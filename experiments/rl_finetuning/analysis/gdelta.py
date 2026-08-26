"""Measure the return term g_delta of the return-weighted ELBO decomposition.

MiniHack port of the sibling repository's
``experiments/rl_finetuning/analysis/gdelta.py``. Same output JSON schema and
same reported columns; the differences are PyTorch for JAX and one
environment-forced flag mapping, noted under ``num_envs`` below.

Writing the per-window weight as A_i and the batch mean as Abar, the training
gradient decomposes exactly as

    grad L_RW  =  Abar * ( grad L_BC  +  g_delta ),
    g_delta    =  (1/B) sum_i delta_i grad l_i,     delta_i = A_i/Abar - 1,

so g_delta carries the entire directional contribution of the return and Abar
is the scalar step-size rescaling. :func:`measure` loads a pretrained
checkpoint, collects one on-policy batch from it, and evaluates grad L_BC,
g_delta and grad L_RW on that batch at those parameters under a shared
(z_t, t) draw, so the only difference between the three is the weight vector.
It repeats for every weight transform the ablation suite uses, and reports
three references the cosine column needs:

  * the random-direction null, cos ~ N(0, 1/sqrt(D)) for D parameters;
  * cos(grad L_BC, grad L_BC) across two independent noise draws, which is the
    value a direction attains when it *is* the imitation direction;
  * the shuffled-delta null, which permutes delta across the batch. This keeps
    the multiset of weights and hence CV_A, and destroys the association
    between a window's weight and that window's gradient. Any part of the
    measured ratio and cosine that survives the shuffle is batch
    heterogeneity, not return signal.

L_BC and L_RW here are the (weighted) mean per-window NELBO alone. The
auxiliary goal-prediction term the trainer adds is not weighted by the
advantages, so including it would break the identity above at the third
decimal for reasons that have nothing to do with the return. The decomposition
is a statement about the ELBO term, and that is what is measured. One
consequence: the auxiliary goal head receives no gradient here, so its
coordinates are identically zero in all three vectors. They are still counted
in ``D``, which makes the reported random-direction null marginally
conservative.

The shared ``(z_t, t)`` draw is a shared PRNG seed rather than a shared key:
every gradient evaluation reseeds the global torch generator immediately before
sampling t and masking, so the three weight vectors see the identical draw.

No training and no optimiser step occur. Runs on CPU in a few minutes.

The per-draw standard deviations a single seed reports are *within* one rollout
seed. The figure the paper's gradient-decomposition table prints is the
standard deviation across rollout seeds, which :func:`aggregate` computes over
the per-seed records.

Driven by ``run_ablations.py --measure-gdelta``; see :func:`run_gdelta`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import orjson

logger = logging.getLogger(__name__)

# The variant -> ablation mapping this module assumes. Each entry names the
# registry key, the loss factory that key must still use, and its wins_only
# flag. verify_registry() fails loudly if the suite drifts away from it, so a
# registry edit cannot silently desynchronise the measurement.
REGISTRY_RULES = {
    "baseline_clipped_ratio": ("baseline_rl", "make_loss_baseline", False),
    "advantage_clip": ("advantage_clip", "make_loss_advantage_clip", False),
    "normalized_adv": ("normalized_adv", "make_loss_normalized_adv", False),
    "bc_wins": ("bc_wins", "make_loss_bc_wins", True),
}

BASELINE = "baseline_clipped_ratio"

GDELTA_DIRNAME = "gdelta"
AGGREGATE_FILENAME = "gdelta_aggregate.json"


class RegistryDriftError(RuntimeError):
    """The ablation registry no longer matches the variants measured here."""


def verify_registry() -> None:
    """Fail if the ablation registry no longer matches the assumed variants."""
    from experiments.rl_finetuning.ablations.registry import REGISTRY

    for variant, (name, factory, wins_only) in REGISTRY_RULES.items():
        spec = REGISTRY.get(name)
        if spec is None:
            raise RegistryDriftError(
                f"registry has no ablation {name!r}; variant {variant!r} is stale"
            )
        if spec.loss_factory.__name__ != factory:
            raise RegistryDriftError(
                f"ablation {name!r} now uses {spec.loss_factory.__name__}, "
                f"not {factory}; variant {variant!r} measures a weighting the "
                "trainer no longer applies"
            )
        if bool(spec.wins_only) != wins_only:
            raise RegistryDriftError(
                f"ablation {name!r} has wins_only={spec.wins_only}, expected "
                f"{wins_only}; variant {variant!r} is stale"
            )


def build_variants(adv, returns, cfg, batch):
    """The weight vectors the suite's four weighting ablations apply.

    ``bc_wins`` is the binary win mask rescaled by ``B / n_wins``, which is the
    vector ``make_loss_bc_wins`` builds internally from the mask that
    ``compute_advantages(wins_only=True)`` hands it. The rescaling is what
    turns the loss's batch mean into a uniform mean over the winners, so it is
    part of the weight the trainer applies, not a normalisation added here.
    """
    eps = cfg["adv_clip_eps"]
    win = (returns > cfg["win_threshold"]).float()
    n_win = win.sum()
    scale = (batch / n_win.clamp(min=1.0)) * (n_win > 0).float()
    return {
        BASELINE: adv,
        "advantage_clip": adv.clamp(1.0 - eps, 1.0 + eps),
        "normalized_adv": (adv - adv.mean()) / (adv.std() + 1e-8),
        "bc_wins": win * scale,
    }


def centred_delta(weights):
    """Return ``(delta, Abar, a1_holds)`` for a weight vector.

    (A1) of the paper asks for non-negative weights with a strictly positive
    mean. Where it holds, ``delta_i = A_i/Abar - 1`` and ``mean(delta) == 0``.
    Where it does not -- ``normalized_adv`` mean-centres, so its weights are
    signed and its mean vanishes -- the ratio is meaningless, and the caller is
    told rather than handed a number that divides by a value near zero.
    """
    wbar = float(weights.mean())
    scale = float(weights.abs().max()) + 1e-12
    a1_holds = bool((weights >= 0.0).all()) and wbar > 1e-6 * scale
    if not a1_holds:
        return weights - wbar, wbar, False
    return weights / wbar - 1.0, wbar, True


def against_bc(g, g_bc, norm_bc):
    """``(||g|| / ||g_BC||, cos(g, g_BC))`` for one gradient against imitation."""
    import torch

    norm = float(g.norm())
    return norm / norm_bc, float(torch.dot(g, g_bc) / (norm * norm_bc + 1e-12))


def effective_sample_size(weights) -> float:
    """ESS as a fraction of the batch: (sum A)^2 / (B sum A^2)."""
    total = float(weights.sum())
    sq = float((weights**2).sum())
    if sq <= 0.0:
        return float("nan")
    return total**2 / (sq * weights.shape[0])


def restore_params(cfg, ckpt: str, device):
    """Build the model from the config and load the checkpoint's EMA weights.

    The sibling needs an explicit Orbax sharding to restore a GPU-written
    checkpoint on CPU. A ``torch.load`` with ``map_location`` has no such
    problem, so this is the plain library path; the function is kept for
    signature parity with the sibling.
    """
    import torch

    from src.models.denoiser import make_model

    model = make_model(cfg).to(device)
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    state = blob.get("ema_state_dict", blob)
    model.load_state_dict(state)
    step = int(blob.get("iteration", 0)) if isinstance(blob, dict) else 0
    return model, step


def aggregate(blobs: list[dict], inputs: list[str] | None = None) -> dict:
    """Combine per-seed records, reporting the standard deviation across seeds.

    Each input contributes one number per variant per column -- its own mean
    over the ``(z_t, t)`` draws. The dispersion reported here is over those
    per-seed means, which is the quantity the paper's table claims.

    Args:
        blobs:  Per-seed records, as returned by :func:`measure`.
        inputs: Optional provenance paths recorded in the output.

    Returns:
        The aggregate record.

    Raises:
        ValueError: If an input is itself an aggregate, or the variant sets
            disagree.
    """
    labels = inputs if inputs is not None else [
        f"seed{b.get('seed', i)}" for i, b in enumerate(blobs)
    ]
    for blob, label in zip(blobs, labels, strict=True):
        if blob.get("aggregate"):
            raise ValueError(f"{label} is already an aggregate")

    names = list(blobs[0]["variants"])
    for blob, label in zip(blobs, labels, strict=True):
        if list(blob["variants"]) != names:
            raise ValueError(f"{label} has a different variant set")

    def across(values):
        arr = np.array(values, dtype=float)
        return float(arr.mean()), float(arr.std())

    out = {
        "aggregate": True,
        "inputs": [str(p) for p in labels],
        "seeds": [int(b["seed"]) for b in blobs],
        "n_seeds": len(blobs),
        "n_draws_per_seed": [int(b["n_draws"]) for b in blobs],
        "n_params": int(blobs[0]["n_params"]),
        "random_cos_sd": float(blobs[0]["random_cos_sd"]),
        "batch": int(blobs[0]["batch"]),
        "eq4_residual_max": float(max(b["eq4_residual_max"] for b in blobs)),
        "variants": {},
    }
    out["bc_self_cos_mean"], out["bc_self_cos_std_seeds"] = across(
        [b["bc_self_cos_mean"] for b in blobs]
    )

    columns = [
        "cv_a", "abar", "abar_ratio_to_baseline", "ess_fraction",
        "ratio_mean", "cos_mean", "ratio_shuffled_mean", "cos_shuffled_mean",
    ]
    for name in names:
        rec = {"a1_violated": any(b["variants"][name]["a1_violated"] for b in blobs)}
        for col in columns:
            mean, std = across([b["variants"][name][col] for b in blobs])
            stem = col[:-5] if col.endswith("_mean") else col
            rec[f"{stem}_mean"] = mean
            rec[f"{stem}_std_seeds"] = std
        out["variants"][name] = rec
    return out


def print_aggregate(agg: dict) -> None:
    print(f"\naggregate over {agg['n_seeds']} rollout seeds {agg['seeds']}, "
          f"{agg['n_draws_per_seed']} draws each; +/- is ACROSS SEEDS")
    print(f"cos(grad L_BC, grad L_BC) = {agg['bc_self_cos_mean']:.3f} "
          f"+/- {agg['bc_self_cos_std_seeds']:.3f}")
    print(f"Eq. 4 identity, max relative residual = {agg['eq4_residual_max']:.2e}\n")
    header = (f"{'weight transform':26s} {'CV_A':>7s} {'Abar':>8s} {'Abar/base':>10s} "
              f"{'ESS/B':>7s} {'ratio':>16s} {'cos':>16s} "
              f"{'ratio(shuf)':>16s} {'cos(shuf)':>16s}")
    print(header)
    for name, rec in agg["variants"].items():
        flag = "  [(A1) violated]" if rec["a1_violated"] else ""
        print(f"{name:26s} {rec['cv_a_mean']:7.3f} {rec['abar_mean']:8.3f} "
              f"{rec['abar_ratio_to_baseline_mean']:10.3f} "
              f"{rec['ess_fraction_mean']:7.3f} "
              f"{rec['ratio_mean']:9.3f} +/-{rec['ratio_std_seeds']:.3f} "
              f"{rec['cos_mean']:+9.3f} +/-{rec['cos_std_seeds']:.3f} "
              f"{rec['ratio_shuffled_mean']:9.3f} +/-{rec['ratio_shuffled_std_seeds']:.3f} "
              f"{rec['cos_shuffled_mean']:+9.3f} +/-{rec['cos_shuffled_std_seeds']:.3f}"
              f"{flag}")


def measure(
    config: dict,
    ckpt: str,
    *,
    seed: int = 0,
    n_draws: int = 8,
    num_envs: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
) -> dict:
    """Measure the decomposition at one rollout seed.

    Args:
        config:     Lowercase config dict; the run's own, so the weight
                    transforms match the ones it trained under. It must carry
                    the structural keys ``make_model`` needs as well as the
                    scalar ones, which a bare ``results.json`` config does not.
        ckpt:       Checkpoint ``.pth`` of the pretrained planner.
        seed:       Rollout seed. Also fixes the ``(z_t, t)`` draws.
        n_draws:    Independent ``(z_t, t)`` draws to average over.
        num_envs:   Override the collection size; on MiniHack, whose rollouts
                    are sequential rather than vectorised, the sibling's
                    ``NUM_ENVS`` has no counterpart and this sets
                    ``episodes_per_iter``. ``None`` keeps the config value.
        batch_size: Override ``batch_size``; ``None`` keeps the config value.
        device:     Torch device; ``None`` picks CUDA where available.

    Returns:
        The per-seed record.

    Raises:
        RuntimeError: If collection produced no windows.
    """
    import random

    import torch

    from experiments.rl_finetuning.ablations.losses import _forward_and_loss
    from experiments.rl_finetuning.ablations.training import (
        collect_training_data,
        compute_advantages,
    )
    from src.diffusion.schedules import get_schedule

    verify_registry()

    raw = dict(config)
    if num_envs is not None:
        raw["episodes_per_iter"] = num_envs
    if batch_size is not None:
        raw["batch_size"] = batch_size
    cfg = SimpleNamespace(**raw)
    cfg.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_t = torch.device(cfg.device)
    cfg._schedule_fn = get_schedule(cfg.noise_schedule)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, step = restore_params(cfg, ckpt, device_t)
    n_params = sum(p.numel() for p in model.parameters())
    random_cos_sd = 1.0 / np.sqrt(n_params)
    print(f"checkpoint step {step}, D = {n_params/1e6:.2f}M, "
          f"random-cosine null sd = {random_cos_sd:.2e}", flush=True)

    # ---- one on-policy batch from the pretrained policy ----
    model.eval()
    local_obs, global_obs, x0, returns = collect_training_data(
        model, cfg, device_t, cfg.episodes_per_iter
    )
    if local_obs.shape[0] == 0:
        raise RuntimeError("collection produced no windows")
    adv, _, _ = compute_advantages(
        returns,
        cfg.return_weight_floor,
        cfg.return_weight_cap,
        wins_only=False,
        win_thresh=cfg.win_threshold,
        use_running_stats=False,
        ema_decay=0.99,
        running_mean=0.0,
        running_std=1.0,
    )

    batch = min(cfg.batch_size, local_obs.shape[0])
    idx = torch.randperm(local_obs.shape[0], device=device_t)[:batch]
    local_b, global_b, x0_b = local_obs[idx], global_obs[idx], x0[idx]
    adv_b, ret_b = adv[idx], returns[idx]
    print(f"batch {batch} windows, win rate "
          f"{float((ret_b > cfg.win_threshold).float().mean()):.3f}", flush=True)

    variants = build_variants(adv_b, ret_b, raw, batch)

    params = [p for p in model.parameters() if p.requires_grad]
    model.train()

    def gradient(weights, draw_seed: int):
        """Flat gradient of the (weighted) mean per-window NELBO.

        Reseeding here is what makes the ``(z_t, t)`` draw shared: two calls
        with the same ``draw_seed`` differ only in ``weights``.
        """
        torch.manual_seed(draw_seed)
        model.zero_grad(set_to_none=True)
        per_sample, _, _, _, _ = _forward_and_loss(
            model, local_b, global_b, x0_b, cfg, device_t
        )
        loss = per_sample.mean() if weights is None else (per_sample * weights).mean()
        loss.backward()
        return torch.cat([
            (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
            for p in params
        ]).detach()

    deltas = {}
    stats = {}
    for name, weights in variants.items():
        delta, wbar, a1_holds = centred_delta(weights)
        deltas[name] = delta
        stats[name] = {
            "cv_a": float(torch.sqrt((delta**2).mean())),
            "abar": wbar,
            "ess_fraction": effective_sample_size(weights),
            "a1_violated": not a1_holds,
        }
    base_abar = stats[BASELINE]["abar"]
    for rec in stats.values():
        rec["abar_ratio_to_baseline"] = rec["abar"] / base_abar

    acc = {name: {"ratio": [], "cos": [], "ratio_shuf": [], "cos_shuf": []}
           for name in variants}
    bc_self, residuals = [], []
    # A separate stream for the shuffled-delta null, so adding the control
    # leaves every draw of the real measurement bit-for-bit unchanged.
    perm_gen = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    for draw in range(n_draws):
        key = seed * 1_000_003 + draw
        g_bc = gradient(None, key)
        norm_bc = float(g_bc.norm())

        g_bc2 = gradient(None, key + 500_000)  # same objective, independent draw
        bc_self.append(
            float(torch.dot(g_bc, g_bc2) / (norm_bc * float(g_bc2.norm()) + 1e-12))
        )

        for name, weights in variants.items():
            delta = deltas[name]
            g_delta = gradient(delta, key)
            ratio, cos = against_bc(g_delta, g_bc, norm_bc)
            acc[name]["ratio"].append(ratio)
            acc[name]["cos"].append(cos)

            # Shuffled-delta null: same multiset of weights, same CV_A, the
            # association with each window's own return destroyed. The (z_t, t)
            # seed is the one the real measurement used, so the two differ only
            # in which window carries which weight.
            perm = torch.randperm(delta.shape[0], generator=perm_gen).to(delta.device)
            ratio_s, cos_s = against_bc(gradient(delta[perm], key), g_bc, norm_bc)
            acc[name]["ratio_shuf"].append(ratio_s)
            acc[name]["cos_shuf"].append(cos_s)

            if name == BASELINE:
                # Eq. 4 identity check: grad L_RW == Abar * (grad L_BC + g_delta)
                g_rw = gradient(weights, key)
                residuals.append(float(
                    (g_rw - stats[name]["abar"] * (g_bc + g_delta)).norm()
                    / (g_rw.norm() + 1e-12)
                ))
        print(f"  draw {draw + 1}/{n_draws}", flush=True)

    out = {
        "aggregate": False,
        "checkpoint_step": int(step),
        "n_params": int(n_params),
        "random_cos_sd": float(random_cos_sd),
        "batch": int(batch),
        "seed": seed,
        "n_draws": n_draws,
        "bc_self_cos_mean": float(np.mean(bc_self)),
        "bc_self_cos_std": float(np.std(bc_self)),
        "eq4_residual_max": float(np.max(residuals)),
        "variants": {},
    }
    print(f"\ncos(grad L_BC, grad L_BC) across draws = "
          f"{np.mean(bc_self):.3f} +/- {np.std(bc_self):.3f}   [same-objective reference]")
    print(f"random-direction null: cos ~ N(0, {random_cos_sd:.2e})")
    print(f"Eq. 4 identity, max relative residual = {np.max(residuals):.2e}")
    print(f"+/- below is across the {n_draws} (z_t, t) draws of this one seed, "
          "not across seeds\n")
    print(f"{'weight transform':26s} {'CV_A':>7s} {'Abar':>8s} {'Abar/base':>10s} "
          f"{'ESS/B':>7s} {'ratio':>16s} {'cos':>16s} "
          f"{'ratio(shuf)':>16s} {'cos(shuf)':>16s}")
    for name, rec in acc.items():
        ratio, cos = np.array(rec["ratio"]), np.array(rec["cos"])
        ratio_s, cos_s = np.array(rec["ratio_shuf"]), np.array(rec["cos_shuf"])
        out["variants"][name] = {
            **stats[name],
            "ratio_mean": float(ratio.mean()), "ratio_std_draws": float(ratio.std()),
            "cos_mean": float(cos.mean()), "cos_std_draws": float(cos.std()),
            "ratio_shuffled_mean": float(ratio_s.mean()),
            "ratio_shuffled_std": float(ratio_s.std()),
            "cos_shuffled_mean": float(cos_s.mean()),
            "cos_shuffled_std": float(cos_s.std()),
        }
        flag = "  [(A1) violated]" if stats[name]["a1_violated"] else ""
        print(f"{name:26s} {stats[name]['cv_a']:7.3f} {stats[name]['abar']:8.3f} "
              f"{stats[name]['abar_ratio_to_baseline']:10.3f} "
              f"{stats[name]['ess_fraction']:7.3f} "
              f"{ratio.mean():9.3f} +/-{ratio.std():.3f} "
              f"{cos.mean():+9.3f} +/-{cos.std():.3f} "
              f"{ratio_s.mean():9.3f} +/-{ratio_s.std():.3f} "
              f"{cos_s.mean():+9.3f} +/-{cos_s.std():.3f}{flag}")
    return out


def _write(path: Path, blob: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(blob, option=orjson.OPT_INDENT_2))
    logger.info("Wrote %s", path)


def load_gdelta_aggregate(output_dir: Path) -> dict | None:
    """The aggregate record for a run, or ``None`` if it was never measured."""
    path = Path(output_dir) / GDELTA_DIRNAME / AGGREGATE_FILENAME
    if not path.is_file():
        return None
    return orjson.loads(path.read_bytes())


def run_gdelta(
    config: dict,
    ckpt: str,
    output_dir: Path,
    *,
    seeds: list[int],
    n_draws: int = 8,
    num_envs: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
) -> dict:
    """Measure every seed and write the per-seed and aggregate records.

    Artifacts land in ``output_dir/gdelta/``, beside the run's ``results.json``,
    so one run directory holds the suite's scores and the gradient measurement
    taken at the checkpoint they started from.

    Args:
        config:     Lowercase config dict.
        ckpt:       Checkpoint ``.pth`` of the pretrained planner.
        output_dir: The run's root output directory.
        seeds:      Rollout seeds to measure.
        n_draws:    Independent ``(z_t, t)`` draws per seed.
        num_envs:   Override the collection size (``episodes_per_iter``).
        batch_size: Override ``batch_size``; ``None`` keeps the config value.
        device:     Torch device; ``None`` picks CUDA where available.

    Returns:
        The aggregate record.
    """
    gdelta_dir = Path(output_dir) / GDELTA_DIRNAME
    blobs, labels = [], []
    for seed in seeds:
        logger.info("Measuring g_delta at rollout seed %d", seed)
        blob = measure(
            config,
            ckpt,
            seed=seed,
            n_draws=n_draws,
            num_envs=num_envs,
            batch_size=batch_size,
            device=device,
        )
        path = gdelta_dir / f"gdelta_seed{seed}.json"
        _write(path, blob)
        blobs.append(blob)
        labels.append(str(path))

    agg = aggregate(blobs, labels)
    _write(gdelta_dir / AGGREGATE_FILENAME, agg)
    print_aggregate(agg)
    return agg


def aggregate_files(paths: list[str], output_dir: Path) -> dict:
    """Aggregate per-seed records written elsewhere, e.g. by another machine."""
    blobs = [orjson.loads(Path(p).read_bytes()) for p in paths]
    agg = aggregate(blobs, [str(p) for p in paths])
    _write(Path(output_dir) / GDELTA_DIRNAME / AGGREGATE_FILENAME, agg)
    print_aggregate(agg)
    return agg
