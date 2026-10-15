"""Signed, stateful region refinement with pixel-level click projection."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from benchmarks.v4.phase_6_mask_decoder.src.model import LocalGraphBlock, knn_adjacency


def pixel_probability(assignment: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    weights = assignment.float() / assignment.float().sum(1, keepdim=True).clamp_min(1e-8)
    return torch.einsum("bkhw,bk->bhw", weights, logits.float().sigmoid())


def click_values(image: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """Read [B,H,W] at normalized click coordinates [B,P,2]."""
    height, width = image.shape[-2:]
    x = (xy[..., 0].clamp(0, 1) * width).long().clamp(max=width - 1)
    y = (xy[..., 1].clamp(0, 1) * height).long().clamp(max=height - 1)
    return image.flatten(1).gather(1, y * width + x)


def refine_pixel_clicks(probability: torch.Tensor, xy: torch.Tensor,
                        sign: torch.Tensor, valid: torch.Tensor,
                        floor: float, ceiling: float,
                        radius: float = 2.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Add a local pixel residual and anchor every valid click exactly.

    A soft region can cover opposite clicks; distinct pixels remain satisfiable
    through this small pixel branch. Opposite signs at the same pixel conflict.
    """
    batch, height, width = probability.shape
    px = (xy[..., 0].clamp(0, 1) * width).long().clamp(max=width - 1)
    py = (xy[..., 1].clamp(0, 1) * height).long().clamp(max=height - 1)
    conflict = torch.zeros(batch, dtype=torch.bool, device=probability.device)
    for first in range(xy.shape[1]):
        for second in range(first + 1, xy.shape[1]):
            conflict |= (valid[:, first] & valid[:, second]
                         & (px[:, first] == px[:, second])
                         & (py[:, first] == py[:, second])
                         & (sign[:, first] != sign[:, second]))
    yy, xx = torch.meshgrid(torch.arange(height, device=probability.device),
                            torch.arange(width, device=probability.device), indexing="ij")
    result = probability
    for point in range(xy.shape[1]):
        target = torch.where(sign[:, point].bool(),
                             torch.full((batch,), floor, device=result.device),
                             torch.full((batch,), ceiling, device=result.device))
        current = result[torch.arange(batch, device=result.device), py[:, point], px[:, point]]
        delta = torch.where(sign[:, point].bool(), (target - current).clamp_min(0),
                            (target - current).clamp_max(0)) * valid[:, point]
        distance = ((xx[None] - px[:, point, None, None]).float().square()
                    + (yy[None] - py[:, point, None, None]).float().square())
        basis = torch.exp(-distance / (2 * radius * radius))
        result = (result + delta[:, None, None] * basis).clamp(0, 1)
    flat = result.flatten(1)
    for point in range(xy.shape[1]):
        index = py[:, point] * width + px[:, point]
        current = flat.gather(1, index[:, None]).squeeze(1)
        target = torch.where(sign[:, point].bool(),
                             torch.full_like(current, floor), torch.full_like(current, ceiling))
        replacement = torch.where(valid[:, point], target, current)
        flat = flat.scatter(1, index[:, None], replacement[:, None])
    return flat.view_as(result), conflict


class IterativePromptDecoder(nn.Module):
    """Update a frozen baseline prediction after one signed click.

    Prompt tokens are never pooled before region matching. The baseline's region
    tokens, logits and soft assignment must describe the same cached geometry.
    Zero update heads preserve prior region logits before the click pixel anchor.
    """

    def __init__(self, dim: int = 256, heads: int = 4, graph_layers: int = 2,
                 neighbours: int = 6, max_delta: float = 4.0):
        super().__init__()
        self.dim = dim
        self.neighbours = neighbours
        self.max_delta = float(max_delta)
        self.sign = nn.Embedding(2, dim)
        self.prompt_xy = nn.Linear(2, dim)
        self.prompt_norm = nn.LayerNorm(dim)
        self.region_norm = nn.LayerNorm(dim)
        self.prompt_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.state = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.input_norm = nn.LayerNorm(dim)
        self.graph = nn.ModuleList(LocalGraphBlock(dim, heads, 0.0) for _ in range(graph_layers))
        self.positive_gain = nn.Linear(dim, 1)
        self.negative_gain = nn.Linear(dim, 1)
        for head in (self.positive_gain, self.negative_gain):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, *, region_tokens: torch.Tensor, region_xy: torch.Tensor,
                region_area: torch.Tensor, active: torch.Tensor,
                prompt_tokens: torch.Tensor, prompt_xy: torch.Tensor,
                prompt_sign: torch.Tensor, prompt_valid: torch.Tensor,
                new_index: torch.Tensor, previous_logits: torch.Tensor,
                assignment: torch.Tensor, click_floor: float = 0.7,
                click_ceiling: float = 0.3, project_clicks: bool = False,
                anchor_xy: torch.Tensor | None = None,
                anchor_valid: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        batch, regions, dim = region_tokens.shape
        if dim != self.dim or previous_logits.shape != (batch, regions):
            raise ValueError("region token/logit shape mismatch")
        if assignment.shape[:2] != (batch, regions) or active.shape != (batch, regions):
            raise ValueError("assignment/active shape mismatch")
        if prompt_tokens.shape[:2] != prompt_xy.shape[:2] or prompt_tokens.shape[:2] != prompt_sign.shape:
            raise ValueError("prompt shape mismatch")
        if prompt_valid.shape != prompt_sign.shape or new_index.shape != (batch,):
            raise ValueError("prompt mask/new index shape mismatch")
        if (new_index < 0).any() or (new_index >= prompt_sign.shape[1]).any():
            raise ValueError("new click index outside prompt set")
        new_sign = prompt_sign.gather(1, new_index[:, None]).squeeze(1)
        if not prompt_valid.gather(1, new_index[:, None]).all() or not ((prompt_sign == 0) | (prompt_sign == 1)).all():
            raise ValueError("new click must be valid and signs must be 0/1")
        if not prompt_valid.any(1).all():
            raise ValueError("each episode needs a prompt")
        if not (0 < click_ceiling < click_floor < 1):
            raise ValueError("require 0 < click_ceiling < click_floor < 1")
        anchor_xy = prompt_xy if anchor_xy is None else anchor_xy
        anchor_valid = prompt_valid if anchor_valid is None else anchor_valid
        if anchor_xy.shape != prompt_xy.shape or anchor_valid.shape != prompt_valid.shape:
            raise ValueError("anchor tensors must match prompt shapes")
        if (anchor_valid & ~prompt_valid).any():
            raise ValueError("a local anchor must be a valid semantic prompt")

        prior = previous_logits.detach().float()
        encoded = self.prompt_norm(prompt_tokens + self.prompt_xy(prompt_xy) + self.sign(prompt_sign.long()))
        encoded = encoded * prompt_valid.unsqueeze(-1)
        region = self.region_norm(region_tokens)
        visual_prompt = self.region_norm(prompt_tokens)
        context, _ = self.prompt_attention(region, encoded, encoded,
                                           key_padding_mask=~prompt_valid, need_weights=False)
        sim = F.cosine_similarity(region[:, :, None], visual_prompt[:, None], dim=-1)
        sim = sim.masked_fill(~prompt_valid[:, None], -1.0)
        new_prompt = visual_prompt.gather(1, new_index[:, None, None].expand(-1, 1, dim)).squeeze(1)
        new_sim = F.cosine_similarity(region, new_prompt[:, None], dim=-1)
        state = self.state(torch.stack((prior, region_area.float(), new_sim), -1))
        hidden = self.input_norm(region + context + state)
        active_float = active.to(hidden.dtype)
        hidden = hidden * active_float[..., None]
        adjacency = knn_adjacency(region_xy, active.bool(), self.neighbours)
        for block in self.graph:
            hidden = block(hidden, adjacency) * active_float[..., None]
        # A correction must match the new click more than an opposite-sign
        # support click. The graph changes the strength, never the update sign.
        opposite = prompt_valid & (prompt_sign != new_sign[:, None])
        opposite_sim = sim.masked_fill(~opposite[:, None], -1.0).amax(-1)
        evidence = (new_sim - opposite_sim - 0.05).clamp_min(0).clamp_max(1)
        positive = F.softplus(self.positive_gain(hidden).squeeze(-1)) - F.softplus(torch.zeros_like(prior))
        negative = F.softplus(self.negative_gain(hidden).squeeze(-1)) - F.softplus(torch.zeros_like(prior))
        positive = positive.clamp(0, self.max_delta) * evidence * (1 - prior.sigmoid())
        negative = negative.clamp(0, self.max_delta) * evidence * prior.sigmoid()
        gain = torch.where(new_sign[:, None].bool(), positive, -negative)
        candidate = prior + gain * active_float
        logits = candidate
        region_conflict = torch.zeros(batch, dtype=torch.bool, device=prior.device)
        if project_clicks:
            logits, region_conflict = project_signed_clicks(candidate, assignment, anchor_xy,
                                                             prompt_sign, anchor_valid, active,
                                                             click_floor, click_ceiling)
        region_probability = pixel_probability(assignment, logits)
        probability, pixel_conflict = refine_pixel_clicks(region_probability, anchor_xy,
                                                          prompt_sign, anchor_valid,
                                                          click_floor, click_ceiling)
        return {"logits": logits, "candidate_logits": candidate,
                "region_pixel_probability": region_probability, "pixel_probability": probability,
                "click_probability": click_values(probability, anchor_xy),
                "click_conflict": pixel_conflict, "region_conflict": region_conflict,
                "prompt_similarity": sim, "signed_evidence": evidence,
                "positive_gain": positive, "negative_gain": negative,
                "adjacency": adjacency}


def project_signed_clicks(logits: torch.Tensor, assignment: torch.Tensor,
                          xy: torch.Tensor, sign: torch.Tensor, valid: torch.Tensor,
                          active: torch.Tensor, floor: float, ceiling: float,
                          steps: int = 24) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimal iterative logit correction; report unsatisfied/conflicting clicks.

    Projection is after graph refinement. It acts on all soft slots covering a
    clicked pixel, rather than assuming that the nearest centroid owns it.
    """
    weights = assignment.float() / assignment.float().sum(1, keepdim=True).clamp_min(1e-8)
    height, width = assignment.shape[-2:]
    x = (xy[..., 0].clamp(0, 1) * width).long().clamp(max=width - 1)
    y = (xy[..., 1].clamp(0, 1) * height).long().clamp(max=height - 1)
    click_weights = weights.flatten(2).gather(2, (y * width + x)[:, None].expand(-1, weights.shape[1], -1)).transpose(1, 2)
    click_weights = click_weights * active[:, None].float()
    result = logits.float()
    # Handle each click explicitly; repeated sweeps settle overlapping supports.
    for _ in range(steps):
        for point in range(xy.shape[1]):
            w = click_weights[:, point]
            current = (w * result.sigmoid()).sum(-1)
            target = torch.where(sign[:, point].bool(), torch.full_like(current, floor),
                                 torch.full_like(current, ceiling))
            error = torch.where(sign[:, point].bool(), (target - current).clamp_min(0),
                                (target - current).clamp_max(0))
            error = error * valid[:, point]
            derivative = (w.square() * result.sigmoid() * (1 - result.sigmoid())).sum(-1).clamp_min(1e-4)
            step = (error / derivative).clamp(-2.0, 2.0)
            result = result + step[:, None] * w
    final = (click_weights * result.sigmoid()[:, None]).sum(-1)
    violated = valid & torch.where(sign.bool(), final < floor - 1e-3, final > ceiling + 1e-3)
    return result, violated.any(-1)
