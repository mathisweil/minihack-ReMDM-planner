"""Render the README animation: one evaluation episode, planned and replanned.

Loads the released DAgger checkpoint on CPU and rolls out one evaluation layout
exactly as ``Evaluator._run_episodes_batched`` does for a single episode,
keeping every denoising step of every plan. That evaluator, with its locked
executed prefix, produced the paper's Table 8: the Hub's
``results/inference/eval_online_s0.json``, timestamped 2026-08-21. The episode
shown is the first evaluation layout (0..49) the planner solves in 17 to 80
moves within the frame budget: at least one replan, and at most one fresh plan
after the first 64-move window. This checkpoint's wins are bimodal (a few
moves, or just after a window runs out), so no layout is solved in 17 to 64
moves.

    uv run python scripts/render_rollout_gif.py --out ../mathisweil/assets/minihack-planner.gif
"""

from __future__ import annotations

import argparse
import functools
import subprocess
import sys
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFont, features

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from minihack.tiles import GlyphMapper
from safetensors.torch import load_file

from src.config import load_config
from src.diffusion.sampling import LockedPrefix, remdm_sample
from src.envs.minihack_env import AdvancedObservationEnv, borrow_env
from src.models.denoiser import make_model

# Chrome block shared with sibling GIF renderers; keep identical.
W, H, M, FPS = 840, 480, 24, 10
BG, INK, MUTED, FAINT = "#0d1117", "#e6edf3", "#8b949e", "#6e7681"
RULE, CELL, ACCENT, HARM = "#30363d", "#21262d", "#3987e5", "#e66767"
GAIN = ACCENT
CONTENT = (24, 80, 816, 416)
TOKEN_FILLS = (RULE, MUTED, ACCENT, INK, FAINT)
FONTS = {
    "serif": ("/usr/share/fonts/stix-fonts/STIX2Text-Regular.otf", "STIXGeneral.ttf"),
    "italic": (
        "/usr/share/fonts/stix-fonts/STIX2Text-Italic.otf",
        "STIXGeneralItalic.ttf",
    ),
    "label": (
        "/usr/share/fonts/adobe-source-code-pro/SourceCodePro-Medium.otf",
        "DejaVuSansMono.ttf",
    ),
    "value": (
        "/usr/share/fonts/adobe-source-code-pro/SourceCodePro-Regular.otf",
        "DejaVuSansMono.ttf",
    ),
}


@functools.cache
def font(role: str, size: int) -> ImageFont.FreeTypeFont:
    path, fallback = FONTS[role]
    if not Path(path).exists():
        import matplotlib  # type: ignore[import-not-found, unused-ignore]

        path = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / fallback)
    raqm = features.check("raqm")
    layout = ImageFont.Layout.RAQM if raqm else ImageFont.Layout.BASIC
    return ImageFont.truetype(path, size, layout_engine=layout)


def caps(
    d: ImageDraw.ImageDraw,
    x: float,
    y: int,
    text: str,
    *,
    right: bool = False,
    fill: str = MUTED,
    size: int = 13,
    tracking: int = 1,
) -> float:
    f = font("label", size)
    text = text.upper()
    width = sum(f.getlength(c) for c in text) + tracking * (len(text) - 1)
    if right:
        x -= width
    for c in text:
        d.text((x, y), c, font=f, fill=fill, anchor="ls")
        x += f.getlength(c) + tracking
    return width


def value(
    d: ImageDraw.ImageDraw,
    x: float,
    y: int,
    text: str,
    *,
    right: bool = False,
    size: int = 20,
    fill: str = INK,
) -> None:
    anchor = "rs" if right else "ls"
    d.text((x, y), text, font=font("value", size), fill=fill, anchor=anchor)


def headline(
    d: ImageDraw.ImageDraw, runs: list[tuple[str, bool]], max_width: float
) -> None:
    fonts = [font("italic" if italic else "serif", 28) for _, italic in runs]
    widths = [f.getlength(text) for (text, _), f in zip(runs, fonts, strict=True)]
    if sum(widths) > max_width:
        raise ValueError(f"headline is {sum(widths):.0f} px, over {max_width:.0f}")
    x: float = M
    for (text, _), f, w in zip(runs, fonts, widths, strict=True):
        d.text((x, 48), text, font=f, fill=INK, anchor="ls")
        x += w


def hairline(
    d: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int, fill: str = RULE
) -> None:
    d.line([(x0, y0), (x1, y1)], fill=fill, width=1)


def corner_ticks(
    d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], length: int = 6
) -> None:
    x0, y0, x1, y1 = box
    for x, y, sx, sy in (
        (x0, y0, 1, 1),
        (x1, y0, -1, 1),
        (x0, y1, 1, -1),
        (x1, y1, -1, -1),
    ):
        hairline(d, x, y, x + sx * (length - 1), y)
        hairline(d, x, y, x, y + sy * (length - 1))


def chrome(
    runs: list[tuple[str, bool]], kicker: str, foot_left: str, foot_right: str
) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    kicker_width = caps(d, W - M, 48, kicker, right=True)
    headline(d, runs, W - 3 * M - kicker_width)
    hairline(d, M, 64, W - M, 64)
    hairline(d, M, 432, W - M, 432)
    footer = caps(d, M, 456, foot_left) + caps(d, W - M, 456, foot_right, right=True)
    if footer > W - 3 * M:
        raise ValueError(f"footer is {footer:.0f} px, over {W - 3 * M}")
    return img


def upscale(rgb: np.ndarray, k: int) -> Image.Image:
    black = (rgb == 0).all(-1, keepdims=True)
    bg = np.array(ImageColor.getrgb(BG), dtype=np.uint8)
    img = Image.fromarray(np.where(black, bg, rgb).astype(np.uint8))
    return img.resize((img.width * k, img.height * k), Image.Resampling.NEAREST)


def token_strip(
    d: ImageDraw.ImageDraw,
    x: int,
    y: int,
    states: Sequence[int],
    pitch: int = 12,
    cell: tuple[int, int] = (10, 20),
) -> None:
    for i, s in enumerate(states):
        x0 = x + i * pitch
        d.rectangle((x0, y, x0 + cell[0] - 1, y + cell[1] - 1), fill=TOKEN_FILLS[s])


def encode_gif(
    frames: Iterable[Image.Image], out: Path, colours: int, max_bytes: int
) -> int:
    graph = (
        f"[0:v]split[a][b];[a]palettegen=max_colors={colours}:stats_mode=full[p];"
        "[b][p]paletteuse=dither=none:diff_mode=rectangle"
    )
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{W}x{H}", "-framerate", str(FPS), "-i", "-",
        "-filter_complex", graph, "-loop", "0", "-final_delay", "250", str(out),
    ]  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = b"".join(frame.convert("RGB").tobytes() for frame in frames)
    subprocess.run(cmd, input=raw, check=True)
    size = out.stat().st_size
    if size > max_bytes:
        raise ValueError(f"{out} is {size:,} B, over the {max_bytes:,} B cap")
    return size


REPO = "mathisweil/remdm-minihack-checkpoints"
CKPT = "checkpoints/online/Minihack-Online-Diffusion-DAgger-100M"
# Paper Table 8: the checkpoint's win rate on each environment shown, and the
# mean over the four in-distribution environments.
ENV_WIN_RATES = {
    "MiniHack-Room-Random-15x15-v0": "38%",
    "MiniHack-Room-Random-5x5-v0": "82%",
}
ID_WIN_RATE = "48.5%"
STONE, TILE = 2359, 16
# Room actions in NLE CompassDirection order; tokens 8-11 wrap modulo 8.
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1), (-1, 1), (1, 1), (1, -1), (-1, -1))
POSTER_HOLD, MOTION_MAX, COLOURS, MAX_BYTES = 5, 90, 64, 1_500_000
MAP_AREA = (24, 80, 408, 416)
RIGHT, STRIP_W = 432, 31 * 12 + 10
COUNTERS = (RIGHT, RIGHT + 128, RIGHT + 256)


@dataclass(frozen=True)
class Plan:
    rows: list[np.ndarray]
    locked: int
    step: int


@dataclass(frozen=True)
class Episode:
    env_id: str
    ep: int
    states: list[tuple[np.ndarray, np.ndarray]]
    actions: list[int]
    positions: list[int]
    plans: list[Plan]
    won: bool


class Shot(NamedTuple):
    t: int
    plan: int
    k: int
    prev_k: int | None
    now: int | None


class Rows(NamedTuple):
    label: int
    value: int
    plan: int
    strip: int
    legend: int
    note: int


class View(NamedTuple):
    box: tuple[int, int, int, int]
    k: int
    x: int
    y: int
    mask: int
    replan_every: int
    rows: Rows


def load_planner(
    checkpoint_dir: Path | None,
) -> tuple[torch.nn.Module, SimpleNamespace]:
    if checkpoint_dir is None:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(REPO, f"{CKPT}/config.yaml")
        weights = hf_hub_download(REPO, f"{CKPT}/model.safetensors")
    else:
        config_path = checkpoint_dir / "config.yaml"
        weights = checkpoint_dir / "model.safetensors"
    cfg = load_config(str(config_path), {"device": "cpu"})
    model = make_model(cfg)
    model.load_state_dict(load_file(str(weights)))
    return model.eval(), cfg


def eval_seed(env_id: str, ep: int) -> int:
    return 42 + zlib.crc32(f"{env_id}:{ep}".encode()) % (2**31)


def snapshot(env) -> tuple[np.ndarray, np.ndarray]:
    obs = env.last_raw_obs
    return obs["glyphs"].copy(), obs["chars"].copy()


def draft(
    model: torch.nn.Module,
    cfg: SimpleNamespace,
    prefix: LockedPrefix,
    local: np.ndarray,
    glb: np.ndarray,
    step: int,
) -> Plan:
    row = np.array([0])
    prefix.start_window(row)
    history, hist_len = prefix.as_tensors(row, "cpu")
    seq, path, _, _ = remdm_sample(
        model,
        torch.from_numpy(local[None]).long(),
        torch.from_numpy(glb[None]).long(),
        cfg,
        "cpu",
        physics_aware=cfg.physics_aware_sampling,
        history=history,
        hist_len=hist_len,
        return_analytics=True,
    )
    locked = int(hist_len[0])
    first = np.where(
        np.arange(cfg.seq_len) < locked, history[0].numpy(), cfg.mask_token
    )
    return Plan([first, *path[:-1], seq[0].numpy()], locked, step)


@torch.no_grad()
def rollout(
    model: torch.nn.Module, cfg: SimpleNamespace, env_id: str, ep: int, max_steps: int
) -> Episode:
    prefix = LockedPrefix(1, cfg.seq_len, cfg.mask_token)
    actions: list[int] = []
    positions: list[int] = []
    plans: list[Plan] = []
    won, replan, since = False, True, 0
    with borrow_env(env_id, None, cfg) as env:
        (local, glb), _ = env.reset(seed=eval_seed(env_id, ep))
        states = [snapshot(env)]
        for _ in range(max_steps):
            if replan:
                plans.append(draft(model, cfg, prefix, local, glb, len(actions)))
                since = 0
            p = int(prefix.hist_len[0])
            action = max(0, min(int(plans[-1].rows[-1][p]), cfg.action_dim - 1))
            prefix.record(0, action)
            since += 1
            replan = since >= cfg.replan_every or prefix.is_full(0)
            actions.append(action)
            positions.append(p)
            (local, glb), _, term, trunc, info = env.step(action)
            states.append(snapshot(env))
            won = won or bool(info.get("won", False))
            if term or trunc:
                break
    return Episode(env_id, ep, states, actions, positions, plans, won)


def rejection(run: Episode, min_steps: int) -> str | None:
    moves = len(run.actions)
    if not run.won:
        return f"no win in {moves} moves"
    if moves < min_steps:
        return f"won in {moves} moves (under {min_steps}, skipped)"
    if len(shots(run, len(run.plans[0].rows) - 1, 1)) > MOTION_MAX:
        return f"won in {moves} moves (over {MOTION_MAX} frames, skipped)"
    return None


def first_win(
    model: torch.nn.Module,
    cfg: SimpleNamespace,
    env_id: str,
    episodes: range,
    min_steps: int,
    max_steps: int,
) -> Episode:
    for ep in episodes:
        torch.manual_seed(ep)
        run = rollout(model, cfg, env_id, ep, max_steps)
        reason = rejection(run, min_steps)
        outcome = reason or f"won in {len(run.actions)} moves"
        print(f"ep {ep:2d}  seed {eval_seed(env_id, ep):10d}  {outcome}")
        if reason is None:
            return run
    raise RuntimeError(
        f"no layout in {episodes.start}..{episodes.stop - 1} won in "
        f"{min_steps}-{max_steps} moves within {MOTION_MAX} frames"
    )


def map_box(states: list[tuple[np.ndarray, np.ndarray]]) -> tuple[int, int, int, int]:
    seen = np.any([glyphs != STONE for glyphs, _ in states], axis=0)
    ys, xs = np.nonzero(seen)
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def map_scale(box: tuple[int, int, int, int]) -> int:
    y0, x0, y1, x1 = box
    k = min(
        (MAP_AREA[2] - MAP_AREA[0]) // (TILE * (x1 - x0)),
        (MAP_AREA[3] - MAP_AREA[1]) // (TILE * (y1 - y0)),
    )
    if k == 0:
        raise ValueError(f"a {x1 - x0}x{y1 - y0} map does not fit at scale 1")
    return k


def agent_cell(chars: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(chars == ord("@"))
    return int(ys[0]), int(xs[0])


def route(
    chars: np.ndarray, start: tuple[int, int], tokens: np.ndarray, mask: int
) -> list[tuple[int, int]]:
    y, x = start
    cells = []
    for token in tokens:
        if token == mask:
            break
        dy, dx = MOVES[int(token) % len(MOVES)]
        if chars[y + dy, x + dx] not in AdvancedObservationEnv._UNWALKABLE:
            y, x = y + dy, x + dx
        if chars[y, x] == ord(">"):
            break
        cells.append((y, x))
    return cells


def token_states(
    row: np.ndarray, prev: np.ndarray | None, done: int, now: int | None, mask: int
) -> np.ndarray:
    s = np.where(row == mask, 0, 1)
    if prev is not None:
        s[(prev == mask) & (row != mask)] = 2
    s[:done] = 4
    if now is not None:
        s[now] = 3
    return s


def shots(ep: Episode, K: int, f: int) -> list[Shot]:
    out: list[Shot] = []
    ends = [p.step for p in ep.plans[1:]] + [len(ep.actions)]
    for j, (plan, end) in enumerate(zip(ep.plans, ends, strict=True)):
        ks = range(K + 1) if j == 0 else (0, K // 2, K)
        prev = None
        for k in ks:
            out.append(Shot(plan.step, j, k, prev, None))
            prev = k
        for i in range(plan.step, end):
            out += [Shot(i + 1, j, K, None, ep.positions[i])] * f
    return out


def schedule(ep: Episode, K: int) -> list[Shot]:
    motion = shots(ep, K, 2)
    if len(motion) > MOTION_MAX:
        motion = shots(ep, K, 1)
    if len(motion) > MOTION_MAX:
        raise ValueError(
            f"{len(motion)} motion frames, over {MOTION_MAX}: lower --max-steps"
        )
    poster = Shot(len(ep.actions), len(ep.plans) - 1, K, None, ep.positions[-1])
    return [poster] * POSTER_HOLD + motion + [poster]


def view_for(ep: Episode, mask: int, replan_every: int) -> View:
    box = map_box(ep.states)
    k = map_scale(box)
    w, h = (box[3] - box[1]) * TILE * k, (box[2] - box[0]) * TILE * k
    x = MAP_AREA[0] + (MAP_AREA[2] - MAP_AREA[0] - w) // 2
    y = MAP_AREA[1] + (MAP_AREA[3] - MAP_AREA[1] - h) // 2
    return View(box, k, x, y, mask, replan_every, column(y, y + h))


def column(top: int, bottom: int) -> Rows:
    # Counters, plan block and note span the map's height with equal gaps;
    # 145 px is their fixed height from cap top to last baseline.
    gap = (bottom - top - 145) // 2
    plan = top + 46 + gap
    return Rows(top + 9, top + 37, plan, plan + 12, plan + 90, bottom)


def legend(d: ImageDraw.ImageDraw, x: float, y: int) -> float:
    items = (("masked", 0), ("new", 2), ("planned", 1), ("now", 3), ("done", 4))
    for i, (label, state) in enumerate(items):
        if i:
            x += 16
        d.rectangle((x, y - 9, x + 9, y), fill=TOKEN_FILLS[state])
        x += 16 + caps(d, x + 16, y, label)
    return x


def base_frame(ep: Episode, view: View, seq_len: int) -> Image.Image:
    label = ep.env_id.removeprefix("MiniHack-").removesuffix("-v0")
    img = chrome(
        [("Sixty-four moves, drafted ", False), ("all at once", True)],
        "NeurIPS 2026 · BeNTo",
        f"{label} · eval layout {ep.ep}",
        f"wins {ENV_WIN_RATES[ep.env_id]} here · {ID_WIN_RATE} in-distribution",
    )
    d = ImageDraw.Draw(img)
    y0, x0, y1, x1 = view.box
    side = TILE * view.k
    w, h = (x1 - x0) * side, (y1 - y0) * side
    corner_ticks(d, (view.x - 4, view.y - 4, view.x + w + 3, view.y + h + 3))
    rows = view.rows
    for x, counter in zip(COUNTERS, ("step", "plans", "result"), strict=True):
        caps(d, x, rows.label, counter)
    caps(d, RIGHT, rows.plan, f"plan · {seq_len} tokens")
    if legend(d, RIGHT, rows.legend) > RIGHT + STRIP_W:
        raise ValueError("legend overflows the token strip")
    caps(d, RIGHT, rows.note, f"plan {seq_len} · execute {view.replan_every} · replan")
    return img


def map_images(ep: Episode, view: View) -> list[Image.Image]:
    y0, x0, y1, x1 = view.box
    mapper = GlyphMapper()
    return [upscale(mapper.to_rgb(g[y0:y1, x0:x1]), view.k) for g, _ in ep.states]


def draw_chips(
    d: ImageDraw.ImageDraw,
    view: View,
    chips: Sequence[tuple[tuple[int, int], str]],
    size: int,
) -> None:
    side = TILE * view.k
    pad = (side - size) // 2
    for (cy, cx), fill in chips:
        x = view.x + (cx - view.box[1]) * side + pad
        y = view.y + (cy - view.box[0]) * side + pad
        d.rectangle((x - 1, y - 1, x + size, y + size), fill=fill, outline=BG)


def draw_goal(d: ImageDraw.ImageDraw, view: View, cell: tuple[int, int]) -> None:
    side = TILE * view.k
    x = view.x + (cell[1] - view.box[1]) * side
    y = view.y + (cell[0] - view.box[0]) * side
    d.rectangle((x - 1, y - 1, x + side, y + side), outline=INK)


def draw_paths(
    d: ImageDraw.ImageDraw, ep: Episode, shot: Shot, view: View, states: np.ndarray
) -> None:
    stairs = np.argwhere(ep.states[0][1] == ord(">"))[0]
    draw_goal(d, view, (int(stairs[0]), int(stairs[1])))
    chars = ep.states[shot.t][1]
    here = agent_cell(chars)
    trail = [agent_cell(c) for _, c in ep.states[: shot.t]]
    trail_size = TILE * view.k * 3 // 8
    draw_chips(d, view, [(c, FAINT) for c in trail if c != here], trail_size)
    if shot.t == len(ep.actions):
        return
    plan = ep.plans[shot.plan]
    start = plan.locked if shot.now is None else shot.now + 1
    path = route(chars, here, plan.rows[shot.k][start:], view.mask)
    fills = [TOKEN_FILLS[s] for s in states[start:]]
    chips = [(c, f) for c, f in zip(path, fills, strict=False) if c != here]
    draw_chips(d, view, chips, TILE * view.k * 5 // 8)


def draw_panel(
    d: ImageDraw.ImageDraw, ep: Episode, shot: Shot, view: View, states: np.ndarray
) -> None:
    plan = ep.plans[shot.plan]
    rows = view.rows
    won = shot.t == len(ep.actions) and ep.won
    value(d, COUNTERS[0], rows.value, f"{shot.t:03d}")
    value(d, COUNTERS[1], rows.value, f"{shot.plan + 1}")
    value(d, COUNTERS[2], rows.value, "WIN" if won else "–", fill=INK if won else MUTED)
    if shot.now is None:
        status = f"denoise {shot.k:02d}/{len(plan.rows) - 1:02d}"
    else:
        status = f"execute {shot.now - plan.locked + 1:02d}/{view.replan_every}"
    caps(d, RIGHT + STRIP_W, rows.plan, status, right=True)
    half = len(states) // 2
    token_strip(d, RIGHT, rows.strip, states[:half].tolist())
    token_strip(d, RIGHT, rows.strip + 26, states[half:].tolist())


def draw_frame(
    base: Image.Image, maps: list[Image.Image], ep: Episode, shot: Shot, view: View
) -> Image.Image:
    plan = ep.plans[shot.plan]
    prev = None if shot.prev_k is None else plan.rows[shot.prev_k]
    done = plan.locked if shot.now is None else shot.now
    states = token_states(plan.rows[shot.k], prev, done, shot.now, view.mask)
    img = base.copy()
    img.paste(maps[shot.t], (view.x, view.y))
    d = ImageDraw.Draw(img)
    draw_paths(d, ep, shot, view, states)
    draw_panel(d, ep, shot, view, states)
    return img


def underfoot(
    states: list[tuple[np.ndarray, np.ndarray]], cell: tuple[int, int]
) -> tuple[int, int]:
    for glyphs, chars in reversed(states):
        if chars[cell] != ord("@"):
            return glyphs[cell], chars[cell]
    glyphs, chars = states[-1]
    floor = tuple(np.argwhere(chars == ord("."))[0])
    return glyphs[floor], chars[floor]


def arrive(
    states: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    glyphs, chars = states[-1][0].copy(), states[-1][1].copy()
    here = agent_cell(chars)
    stairs = tuple(np.argwhere(chars == ord(">"))[0])
    glyphs[stairs], chars[stairs] = glyphs[here], chars[here]
    glyphs[here], chars[here] = underfoot(states[:-1], here)
    return glyphs, chars


def render(ep: Episode, cfg: SimpleNamespace) -> list[Image.Image]:
    # NLE blanks the observation once the agent takes the stairs.
    ep = replace(ep, states=[*ep.states[:-1], arrive(ep.states[:-1])])
    view = view_for(ep, cfg.mask_token, cfg.replan_every)
    base = base_frame(ep, view, cfg.seq_len)
    maps = map_images(ep, view)
    K = len(ep.plans[0].rows) - 1
    return [draw_frame(base, maps, ep, shot, view) for shot in schedule(ep, K)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--env", choices=list(ENV_WIN_RATES), default=next(iter(ENV_WIN_RATES))
    )
    p.add_argument(
        "--episode", type=int, help="render this layout, skipping the search"
    )
    p.add_argument("--episodes", type=int, default=50, help="layouts searched")
    p.add_argument("--min-steps", type=int, help="default: replan_every + 1")
    p.add_argument("--max-steps", type=int, help="default: seq_len + replan_every")
    p.add_argument(
        "--checkpoint-dir", type=Path, help="local config.yaml + model.safetensors"
    )
    p.add_argument(
        "--out", type=Path, default=_ROOT / "results" / "gif" / "minihack-planner.gif"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model, cfg = load_planner(args.checkpoint_dir)
    if args.episode is None:
        episodes = range(args.episodes)
    else:
        episodes = range(args.episode, args.episode + 1)
    min_steps = args.min_steps or cfg.replan_every + 1
    max_steps = args.max_steps or cfg.seq_len + cfg.replan_every
    run = first_win(model, cfg, args.env, episodes, min_steps, max_steps)
    print(
        f"selected ep {run.ep}: first of {episodes.start}..{episodes.stop - 1} won "
        f"in {min_steps}-{max_steps} moves ({len(run.actions)} moves, "
        f"{len(run.plans)} plans)"
    )
    frames = render(run, cfg)
    size = encode_gif(frames, args.out, COLOURS, MAX_BYTES)
    print(f"{args.out}: {len(frames)} frames, {size:,} B")


if __name__ == "__main__":
    main()
