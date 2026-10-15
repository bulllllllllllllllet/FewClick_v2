"""Apply one click to every prepared tile without recomputing its region encoder."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .episodes import tokens_at_clicks
from .iterative_decoder import IterativePromptDecoder


@dataclass(frozen=True)
class TileState:
    region_tokens: torch.Tensor
    region_xy: torch.Tensor
    region_area: torch.Tensor
    active: torch.Tensor
    assignment: torch.Tensor
    logits: torch.Tensor


@dataclass(frozen=True)
class WholeSlideState:
    slide_id: str
    target_id: str
    cache_version: str
    geometry_version: str
    tiles: dict[str, TileState]
    prompt_tokens: torch.Tensor
    global_prompt_xy: torch.Tensor
    prompt_sign: torch.Tensor
    prompt_tile_id: tuple[str, ...]
    local_prompt_xy: torch.Tensor
    round_index: int = 0


def add_whole_slide_click(decoder: IterativePromptDecoder, state: WholeSlideState,
                          *, slide_id: str, target_id: str, cache_version: str,
                          geometry_version: str, source_tile_id: str,
                          local_xy: torch.Tensor, global_xy: torch.Tensor,
                          sign: int) -> tuple[WholeSlideState, dict]:
    """Use one source click token to update all tiles; anchor only its own tile."""
    if (slide_id, target_id, cache_version, geometry_version) != (
            state.slide_id, state.target_id, state.cache_version, state.geometry_version):
        raise ValueError("whole-slide state identity or cache version mismatch")
    if source_tile_id not in state.tiles or sign not in (0, 1):
        raise ValueError("source tile or sign is invalid")
    if local_xy.shape != (1, 2) or global_xy.shape != (1, 2):
        raise ValueError("click coordinates must each have shape [1,2]")
    source = state.tiles[source_tile_id]
    token = tokens_at_clicks(source.region_tokens, source.assignment, local_xy[:, None])
    tokens = torch.cat((state.prompt_tokens, token), 1)
    global_history = torch.cat((state.global_prompt_xy, global_xy[:, None]), 1)
    local_history = torch.cat((state.local_prompt_xy, local_xy[:, None]), 1)
    sign_history = torch.cat((state.prompt_sign,
                              torch.tensor([[sign]], device=state.prompt_sign.device)), 1)
    tile_history = state.prompt_tile_id + (source_tile_id,)
    valid = torch.ones_like(sign_history, dtype=torch.bool)
    updated = {}
    probabilities = {}
    region_conflicts = {}
    with torch.no_grad():
        for tile_id, tile in state.tiles.items():
            anchors = torch.tensor([owner == tile_id for owner in tile_history],
                                   dtype=torch.bool, device=tile.logits.device)[None]
            result = decoder(region_tokens=tile.region_tokens, region_xy=tile.region_xy,
                             region_area=tile.region_area, active=tile.active,
                             assignment=tile.assignment, previous_logits=tile.logits,
                             prompt_tokens=tokens, prompt_xy=global_history,
                             prompt_sign=sign_history, prompt_valid=valid,
                             new_index=torch.tensor([len(tile_history) - 1], device=tile.logits.device),
                             anchor_xy=local_history, anchor_valid=anchors)
            if result["click_conflict"].item():
                return state, {"status": "abstain", "tile_id": tile_id,
                               "reason": "opposite signed clicks at one pixel"}
            updated[tile_id] = TileState(tile.region_tokens, tile.region_xy, tile.region_area,
                                         tile.active, tile.assignment, result["logits"].detach())
            probabilities[tile_id] = result["pixel_probability"]
            region_conflicts[tile_id] = bool(result["region_conflict"].item())
    next_state = WholeSlideState(state.slide_id, state.target_id, state.cache_version,
                                 state.geometry_version, updated, tokens.detach(),
                                 global_history.detach(), sign_history.detach(), tile_history,
                                 local_history.detach(), state.round_index + 1)
    return next_state, {"status": "ok", "round": next_state.round_index,
                        "tile_probabilities": probabilities,
                        "region_conflicts": region_conflicts}
