"""Versioned in-memory state for consecutive clicks on one cached target."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .episodes import append_click
from .iterative_decoder import IterativePromptDecoder


@dataclass(frozen=True)
class ClickState:
    slide_id: str
    target_id: str
    cache_version: str
    geometry_version: str
    logits: torch.Tensor
    prompt_xy: torch.Tensor
    prompt_sign: torch.Tensor
    prompt_valid: torch.Tensor
    round_index: int = 0


def initial_state(*, slide_id: str, target_id: str, cache_version: str,
                  geometry_version: str, baseline_logits: torch.Tensor,
                  prompt_xy: torch.Tensor, prompt_sign: torch.Tensor,
                  prompt_valid: torch.Tensor) -> ClickState:
    if not all((slide_id, target_id, cache_version, geometry_version)):
        raise ValueError("slide, target, cache and geometry identifiers are required")
    if baseline_logits.shape[0] != 1 or prompt_xy.shape[0] != 1:
        raise ValueError("interactive state expects a single target")
    if not prompt_valid.all():
        raise ValueError("initial history must be compact and valid")
    return ClickState(slide_id, target_id, cache_version, geometry_version,
                      baseline_logits.detach(), prompt_xy.detach(),
                      prompt_sign.detach(), prompt_valid.detach())


@torch.no_grad()
def add_click(decoder: IterativePromptDecoder, state: ClickState, *,
              slide_id: str, target_id: str, cache_version: str,
              geometry_version: str, new_xy: torch.Tensor, new_sign: torch.Tensor,
              region_tokens: torch.Tensor, region_xy: torch.Tensor,
              region_area: torch.Tensor, active: torch.Tensor,
              assignment: torch.Tensor) -> tuple[ClickState, dict]:
    if (slide_id, target_id, cache_version, geometry_version) != (
            state.slide_id, state.target_id, state.cache_version, state.geometry_version):
        raise ValueError("interactive state does not match current slide, target or cache geometry")
    if new_xy.shape != (1, 2) or new_sign.shape != (1,) or new_sign.item() not in (0, 1):
        raise ValueError("one normalized signed click is required")
    packed = append_click(state.prompt_xy, state.prompt_sign, state.prompt_valid,
                          new_xy, new_sign, region_tokens, assignment)
    output = decoder(region_tokens=region_tokens, region_xy=region_xy,
                     region_area=region_area, active=active,
                     previous_logits=state.logits, assignment=assignment, **packed)
    if output["click_conflict"].item():
        return state, {"status": "abstain", "reason": "opposite signed clicks at the same pixel",
                       "round": state.round_index, "pixel_probability": None}
    next_state = ClickState(state.slide_id, state.target_id, state.cache_version,
                            state.geometry_version, output["logits"].detach(),
                            packed["prompt_xy"].detach(), packed["prompt_sign"].detach(),
                            packed["prompt_valid"].detach(), state.round_index + 1)
    return next_state, {"status": "ok", "round": next_state.round_index,
                        "pixel_probability": output["pixel_probability"],
                        "region_conflict": bool(output["region_conflict"].item())}
