import torch

from benchmarks.v5.phase_6_mask_decoder.episodes import (
    append_click, correction_loss, select_correction_click,
)
from benchmarks.v5.phase_6_mask_decoder.iterative_decoder import IterativePromptDecoder, click_values
from benchmarks.v5.phase_6_mask_decoder.interactive import add_click, initial_state
from benchmarks.v5.phase_6_mask_decoder.whole_slide_interactive import TileState, WholeSlideState, add_whole_slide_click


def _scene():
    assignment = torch.zeros(1, 4, 8, 8)
    assignment[:, 0, :4, :4] = 1
    assignment[:, 1, :4, 4:] = 1
    assignment[:, 2, 4:, :4] = 1
    assignment[:, 3, 4:, 4:] = 1
    tokens = torch.randn(1, 4, 16)
    xy = torch.tensor([[[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]])
    return assignment, tokens, xy


def test_signed_clicks_survive_graph_refinement_and_train():
    torch.manual_seed(2)
    assignment, tokens, region_xy = _scene()
    decoder = IterativePromptDecoder(dim=16, heads=4, neighbours=2)
    previous = torch.tensor([[2.0, 2.0, -2.0, -2.0]])
    prompts = torch.tensor([[[0.25, 0.25], [0.75, 0.25], [0.25, 0.75]]])
    signs = torch.tensor([[1, 0, 1]])
    output = decoder(region_tokens=tokens, region_xy=region_xy,
                     region_area=torch.full((1, 4), 0.25),
                     active=torch.ones(1, 4, dtype=torch.bool),
                     prompt_tokens=tokens[:, [0, 1, 2]], prompt_xy=prompts,
                     prompt_sign=signs, prompt_valid=torch.ones(1, 3, dtype=torch.bool),
                     new_index=torch.tensor([2]), previous_logits=previous,
                     assignment=assignment)
    assert not output["click_conflict"].item()
    assert output["click_probability"][0, 0] >= 0.699
    assert output["click_probability"][0, 1] <= 0.301
    assert output["click_probability"][0, 2] >= 0.699
    truth = torch.zeros(1, 8, 8, dtype=torch.bool)
    truth[:, :4, :4] = True
    truth[:, 4:, :4] = True
    prior = assignment[:, 0] * 0.88 + assignment[:, 1] * 0.88 + assignment[:, 2] * 0.12
    loss, _ = correction_loss(output, prior, truth, torch.ones_like(truth),
                              prompts, signs, torch.ones(1, 3, dtype=torch.bool), torch.tensor([2]))
    loss.backward()
    assert torch.isfinite(loss)
    assert decoder.positive_gain.weight.grad is not None
    assert torch.isfinite(decoder.positive_gain.weight.grad).all()


def test_signed_region_logits_are_monotonic_before_pixel_anchor():
    assignment, tokens, region_xy = _scene()
    decoder = IterativePromptDecoder(dim=16, heads=4, graph_layers=0)
    with torch.no_grad():
        decoder.positive_gain.bias.fill_(1.0)
        decoder.negative_gain.bias.fill_(1.0)
    kwargs = dict(region_tokens=tokens, region_xy=region_xy,
                  region_area=torch.full((1, 4), 0.25),
                  active=torch.ones(1, 4, dtype=torch.bool),
                  prompt_tokens=tokens[:, :2],
                  prompt_xy=torch.tensor([[[0.25, 0.25], [0.75, 0.25]]]),
                  prompt_valid=torch.ones(1, 2, dtype=torch.bool),
                  previous_logits=torch.zeros(1, 4), assignment=assignment)
    positive = decoder(**kwargs, prompt_sign=torch.tensor([[0, 1]]), new_index=torch.tensor([1]))
    negative = decoder(**kwargs, prompt_sign=torch.tensor([[1, 0]]), new_index=torch.tensor([1]))
    assert (positive["candidate_logits"] >= -1e-7).all()
    assert (negative["candidate_logits"] <= 1e-7).all()


def test_unresolvable_signed_conflict_is_reported():
    assignment = torch.ones(1, 1, 4, 4)
    decoder = IterativePromptDecoder(dim=8, heads=2, graph_layers=0, neighbours=0)
    output = decoder(region_tokens=torch.randn(1, 1, 8),
                     region_xy=torch.tensor([[[0.5, 0.5]]]),
                     region_area=torch.ones(1, 1), active=torch.ones(1, 1, dtype=torch.bool),
                     prompt_tokens=torch.randn(1, 2, 8),
                     prompt_xy=torch.tensor([[[0.25, 0.25], [0.25, 0.25]]]),
                     prompt_sign=torch.tensor([[1, 0]]), prompt_valid=torch.ones(1, 2, dtype=torch.bool),
                     new_index=torch.tensor([1]), previous_logits=torch.zeros(1, 1),
                     assignment=assignment, project_clicks=True)
    assert output["click_conflict"].item()


def test_distinct_opposite_clicks_in_one_soft_region_use_pixel_residual():
    assignment = torch.ones(1, 1, 8, 8)
    decoder = IterativePromptDecoder(dim=8, heads=2, graph_layers=0, neighbours=0)
    output = decoder(region_tokens=torch.randn(1, 1, 8),
                     region_xy=torch.tensor([[[0.5, 0.5]]]),
                     region_area=torch.ones(1, 1), active=torch.ones(1, 1, dtype=torch.bool),
                     prompt_tokens=torch.randn(1, 2, 8),
                     prompt_xy=torch.tensor([[[0.25, 0.25], [0.75, 0.75]]]),
                     prompt_sign=torch.tensor([[1, 0]]), prompt_valid=torch.ones(1, 2, dtype=torch.bool),
                     new_index=torch.tensor([1]), previous_logits=torch.zeros(1, 1),
                     assignment=assignment, project_clicks=True)
    assert output["region_conflict"].item()
    assert not output["click_conflict"].item()
    assert torch.allclose(output["click_probability"], torch.tensor([[0.7, 0.3]]))


def test_oracle_click_and_prompt_history_use_actual_pixels():
    assignment, tokens, _ = _scene()
    truth = torch.zeros(1, 8, 8, dtype=torch.bool)
    truth[:, 4:, :4] = True
    previous = torch.zeros(1, 8, 8)
    xy, sign = select_correction_click(previous, truth, torch.ones_like(truth), prefer_sign=1)
    packed = append_click(torch.tensor([[[0.25, 0.25]]]), torch.tensor([[0]]),
                          torch.ones(1, 1, dtype=torch.bool), xy, sign, tokens, assignment)
    assert sign.item() == 1
    assert packed["new_index"].item() == 1
    assert click_values(truth.float(), xy[:, None]).item() == 1
    assert torch.allclose(packed["prompt_tokens"][0, 1], tokens[0, 2])


def test_interactive_state_rejects_cache_mismatch_and_conflict():
    assignment, tokens, region_xy = _scene()
    decoder = IterativePromptDecoder(dim=16, heads=4, graph_layers=0)
    state = initial_state(slide_id="s", target_id="t", cache_version="v1",
                          geometry_version="g1", baseline_logits=torch.zeros(1, 4),
                          prompt_xy=torch.tensor([[[0.25, 0.25]]]),
                          prompt_sign=torch.tensor([[1]]),
                          prompt_valid=torch.ones(1, 1, dtype=torch.bool))
    inputs = dict(slide_id="s", target_id="t", cache_version="v1", geometry_version="g1",
                  new_xy=torch.tensor([[0.75, 0.75]]), new_sign=torch.tensor([0]),
                  region_tokens=tokens, region_xy=region_xy,
                  region_area=torch.full((1, 4), 0.25),
                  active=torch.ones(1, 4, dtype=torch.bool), assignment=assignment)
    next_state, response = add_click(decoder, state, **inputs)
    assert response["status"] == "ok" and next_state.round_index == 1
    try:
        add_click(decoder, state, **{**inputs, "cache_version": "v2"})
    except ValueError:
        pass
    else:
        raise AssertionError("cache mismatch was accepted")
    conflict_state, response = add_click(decoder, state,
                                         **{**inputs, "new_xy": torch.tensor([[0.25, 0.25]])})
    assert response["status"] == "abstain" and conflict_state.round_index == 0


def test_whole_slide_click_anchors_only_source_tile():
    assignment, tokens, region_xy = _scene()
    decoder = IterativePromptDecoder(dim=16, heads=4, graph_layers=0)
    tile = TileState(tokens, region_xy, torch.full((1, 4), 0.25),
                     torch.ones(1, 4, dtype=torch.bool), assignment, torch.zeros(1, 4))
    state = WholeSlideState("s", "t", "cache", "geometry", {"a": tile, "b": tile},
                            tokens[:, :1], torch.tensor([[[0.1, 0.1]]]),
                            torch.tensor([[1]]), ("a",), torch.tensor([[[0.25, 0.25]]]))
    next_state, response = add_whole_slide_click(
        decoder, state, slide_id="s", target_id="t", cache_version="cache",
        geometry_version="geometry", source_tile_id="a",
        local_xy=torch.tensor([[0.75, 0.75]]),
        global_xy=torch.tensor([[0.2, 0.2]]), sign=0)
    assert response["status"] == "ok" and next_state.round_index == 1
    source = response["tile_probabilities"]["a"]
    remote = response["tile_probabilities"]["b"]
    assert torch.isclose(click_values(source, torch.tensor([[[0.75, 0.75]]])),
                         torch.tensor([[0.3]])).all()
    assert torch.allclose(remote, torch.full_like(remote, 0.5))
