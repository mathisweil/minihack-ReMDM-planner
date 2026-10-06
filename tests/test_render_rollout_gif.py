"""The README animation replays the evaluation protocol, not a copy of it."""

import importlib.util
import sys
from pathlib import Path

import torch

from tests.conftest import TINY_ENV, requires_minihack

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_rollout_gif.py"


def _renderer():
    spec = importlib.util.spec_from_file_location("render_rollout_gif", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@requires_minihack
def test_rollout_matches_evaluator(tiny_cfg, monkeypatch):
    """A B=1 rollout executes the same actions as the Evaluator under one seed.

    replan_every=4 with seq_len=8 exercises both a locked-prefix replan and a
    fresh window.
    """
    from src.envs.minihack_env import AdvancedObservationEnv
    from src.models.denoiser import make_model
    from src.planners.inference import Evaluator

    gif = _renderer()
    tiny_cfg.replan_every = 4
    model = make_model(tiny_cfg).eval()
    executed = []
    step = AdvancedObservationEnv.step

    def recording_step(self, action):
        executed.append(action)
        return step(self, action)

    monkeypatch.setattr(AdvancedObservationEnv, "step", recording_step)
    torch.manual_seed(0)
    [reference] = Evaluator()._run_episodes_batched(
        model, TINY_ENV, 1, tiny_cfg, "cpu", seeds=[gif.eval_seed(TINY_ENV, 0)]
    )
    monkeypatch.undo()

    torch.manual_seed(0)
    run = gif.rollout(model, tiny_cfg, TINY_ENV, 0, max_steps=500)

    assert run.actions == executed
    assert (len(run.actions), run.won) == (reference["steps"], reference["won"])
    assert {plan.locked for plan in run.plans} == {0, 4}
