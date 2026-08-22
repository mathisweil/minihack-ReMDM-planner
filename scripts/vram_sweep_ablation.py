"""Measure ablation-training peak VRAM against gradient batch size.

The ablation suite slices ``local_obs[:batch_size]`` from whatever the
iteration's episodes yielded, so the gradient batch is data-limited from
above by ``batch_size``. This sweep answers what each attainable batch
costs in VRAM, for the training step alone and for the training step
followed by the diagnostic passes that run on the same batch.

Shapes drive the activation footprint, not glyph values, so the batch is
synthesised rather than collected. The model, loss, AMP setting and
optimizer are the real ones.

Usage:
    python scripts/vram_sweep_ablation.py --batches 1024 2048 3072 4608
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.rl_finetuning.ablations.losses import (  # noqa: E402
    LossContext,
    make_loss_baseline,
)
from experiments.rl_finetuning.diagnostics.gradient import (  # noqa: E402
    compute_grad_alignment,
    compute_per_layer_grad_norms,
)
from experiments.rl_finetuning.diagnostics.representation import (  # noqa: E402
    compute_cka,
    compute_repr_drift,
)
from experiments.rl_finetuning.diagnostics.timestep import (  # noqa: E402
    compute_t_analysis,
)
from experiments.rl_finetuning.run_ablations import (  # noqa: E402
    _load_yaml,
    _merge_to_namespace,
)
from src.diffusion.schedules import get_schedule  # noqa: E402
from src.models.denoiser import make_model  # noqa: E402

MB = 1024.0 * 1024.0


def _batch(cfg, n: int, device: torch.device):
    """Synthesise one gradient batch with the real shapes and dtypes."""
    g = torch.Generator(device="cpu").manual_seed(0)
    local = torch.randint(0, 5999, (n, cfg.crop_size, cfg.crop_size), generator=g)
    glb = torch.randint(0, 5999, (n, cfg.map_h, cfg.map_w), generator=g)
    x0 = torch.randint(0, cfg.action_dim, (n, cfg.seq_len), generator=g)
    adv = torch.rand(n, generator=g)
    return (
        local.to(device),
        glb.to(device),
        x0.to(device),
        adv.to(device),
    )


def _peak(fn) -> tuple[float, str]:
    """Run *fn*, returning (peak MiB allocated, "" or the OOM message)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        fn()
        torch.cuda.synchronize()
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return torch.cuda.max_memory_allocated() / MB, str(exc).split(".")[0]
    return torch.cuda.max_memory_allocated() / MB, ""


def main() -> None:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(_PROJECT_ROOT / "configs/defaults.yaml"))
    p.add_argument(
        "--ablations-config",
        default=str(
            _PROJECT_ROOT / "experiments/rl_finetuning/configs/ablations_final_minihack_gpu_24gb.yaml"
        ),
    )
    p.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[512, 1024, 1536, 2048, 3072, 4608],
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = _merge_to_namespace(
        _load_yaml(args.config), _load_yaml(args.ablations_config)
    )
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    cfg._schedule_fn = get_schedule(cfg.noise_schedule)
    cfg._current_iter = 1

    model = make_model(cfg).to(device)
    ref_model = make_model(cfg).to(device).eval()
    for q in ref_model.parameters():
        q.requires_grad = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, fused=True)
    use_amp = bool(getattr(cfg, "use_amp", False))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = make_loss_baseline(
        LossContext(ref_model=ref_model, schedule_fn=cfg._schedule_fn, cfg=cfg)
    )

    base = torch.cuda.memory_allocated() / MB
    print(f"model + optimizer resident: {base:.0f} MiB")
    print(
        f"{'batch':>7s} {'train':>10s} {'+per_layer':>11s} {'grad_align':>11s} "
        f"{'repr_drift':>11s} {'t_analysis':>11s} {'cka':>9s}"
    )

    rows = []
    for n in args.batches:
        batch = _batch(cfg, n, device)

        def _train(b=batch):
            local, glb, x0, adv = b
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = loss_fn(model, local, glb, x0, adv, cfg, device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

        def _train_then_perlayer(b=batch):
            local, glb, x0, adv = b
            _train(b)
            model.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                dl = loss_fn(model, local, glb, x0, adv, cfg, device)
            dl.backward()
            compute_per_layer_grad_norms(model)
            model.zero_grad()

        m_train, e_train = _peak(_train)
        m_pl, e_pl = _peak(_train_then_perlayer)
        m_ga, e_ga = _peak(
            lambda b=batch: compute_grad_alignment(
                model, ref_model, b[0], b[1], b[2], b[3], cfg, device
            )
        )
        m_rd, e_rd = _peak(
            lambda b=batch: compute_repr_drift(
                model, ref_model, b[0], b[1], b[2], cfg, device
            )
        )
        m_ta, e_ta = _peak(
            lambda b=batch: compute_t_analysis(
                model,
                b[0],
                b[1],
                b[2],
                b[3],
                cfg,
                device,
                n_bins=cfg.t_analysis_n_bins,
            )
        )
        m_ck, e_ck = _peak(
            lambda b=batch: compute_cka(
                model, ref_model, b[0], b[1], b[2], cfg, device
            )
        )

        def _f(m: float, e: str) -> str:
            return f"{m:.0f}{'!' if e else '':>1s}"

        print(
            f"{n:7d} {_f(m_train, e_train):>10s} {_f(m_pl, e_pl):>11s} "
            f"{_f(m_ga, e_ga):>11s} {_f(m_rd, e_rd):>11s} "
            f"{_f(m_ta, e_ta):>11s} {_f(m_ck, e_ck):>9s}"
        )
        rows.append(
            {
                "batch": n,
                "train_mib": m_train,
                "train_oom": bool(e_train),
                "train_plus_per_layer_mib": m_pl,
                "train_plus_per_layer_oom": bool(e_pl),
                "grad_align_mib": m_ga,
                "grad_align_oom": bool(e_ga),
                "repr_drift_mib": m_rd,
                "repr_drift_oom": bool(e_rd),
                "t_analysis_mib": m_ta,
                "t_analysis_oom": bool(e_ta),
                "cka_mib": m_ck,
                "cka_oom": bool(e_ck),
            }
        )
        del batch
        torch.cuda.empty_cache()

    print("\n! marks an OOM; the figure is the peak reached before it.")
    total = torch.cuda.get_device_properties(0).total_memory / MB
    print(f"card total: {total:.0f} MiB")
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "card_total_mib": total,
                    "model_optimizer_mib": base,
                    "n_embd": cfg.n_embd,
                    "use_amp": use_amp,
                    "rows": rows,
                },
                indent=1,
            )
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
