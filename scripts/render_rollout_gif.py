"""Render the README animation: one evaluation episode, drafted and executed.

Loads the released DAgger checkpoint on CPU and rolls out one evaluation layout
exactly as ``Evaluator._run_episodes_batched`` does for a single episode,
keeping every denoising step of every plan. That evaluator, with its locked
executed prefix, produced the paper's Table 8: the Hub's
``results/inference/eval_online_s0.json``, timestamped 2026-08-21. The episode
shown is the first evaluation layout (0..49) the planner wins within its first
plan window of ``replan_every`` moves, so one drafted plan makes every move.
Longer wins mostly show the agent waiting next to the stairs.

    uv run python scripts/render_rollout_gif.py --out ../mathisweil/assets/minihack-planner.gif
"""

from __future__ import annotations

import argparse
import functools
import math
import subprocess
import sys
import zlib
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, features

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

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
    align: str = "left",
    fill: str = MUTED,
    size: int = 13,
    tracking: int = 1,
) -> float:
    f = font("label", size)
    text = text.upper()
    width = sum(f.getlength(c) for c in text) + tracking * (len(text) - 1)
    x -= {"left": 0, "centre": width / 2, "right": width}[align]
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
    align: str = "left",
    size: int = 20,
    fill: str = INK,
) -> None:
    anchor = {"left": "ls", "centre": "ms", "right": "rs"}[align]
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
    kicker_width = caps(d, W - M, 48, kicker, align="right")
    headline(d, runs, W - 3 * M - kicker_width)
    hairline(d, M, 64, W - M, 64)
    hairline(d, M, 432, W - M, 432)
    footer = caps(d, M, 456, foot_left) + caps(d, W - M, 456, foot_right, align="right")
    if footer > W - 3 * M:
        raise ValueError(f"footer is {footer:.0f} px, over {W - 3 * M}")
    return img


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
STONE = 2359
# Room actions in NLE CompassDirection order; tokens 8-11 wrap modulo 8.
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1), (-1, 1), (1, 1), (1, -1), (-1, -1))
# Token states: masked, drafted, new this step, now, done.
TOKEN_FILLS = (CELL, RULE, ACCENT, INK, FAINT)
ARROW_FILLS = (CELL, INK, INK, BG, BG)
ARROW = (
    (-6.5, -1.5),
    (0.5, -1.5),
    (0.5, -5),
    (6.5, 0),
    (0.5, 5),
    (0.5, 1.5),
    (-6.5, 1.5),
)
GLYPH_FILLS = {".": FAINT, "<": INK, ">": INK, "@": INK}
WALKED, AHEAD = (FAINT, 3), (ACCENT, 3)
POSTER_HOLD, DRAFT_HOLD, DENOISE_HOLD, ROUTE_HOLD, MOVE_HOLD = 5, 5, 2, 6, 2
COLOURS, MAX_BYTES = 64, 1_500_000
MAP_AREA, PITCH = (24, 80, 408, 416), 22
RIGHT = 432
COUNTERS = (RIGHT, RIGHT + 128, RIGHT + 256)
LABEL_Y, VALUE_Y, HEAD_Y, LEGEND_Y, NOTE_Y, KEY_Y = 137, 165, 221, 362, 388, 414
TOKEN, TOKEN_PITCH, GRID_COLS = 22, 24, 16
GRID_ROWS = (232, 262, 286, 310)


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
    done: int
    now: int | None
    status: str


class View(NamedTuple):
    box: tuple[int, int, int, int]
    x: int
    y: int
    stairs: tuple[int, int]
    mask: int


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
        f"{min_steps}-{max_steps} moves; raise --max-steps to allow replans"
    )


def map_box(states: list[tuple[np.ndarray, np.ndarray]]) -> tuple[int, int, int, int]:
    seen = np.any([glyphs != STONE for glyphs, _ in states], axis=0)
    ys, xs = np.nonzero(seen)
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


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
        cells.append((y, x))
        if chars[y, x] == ord(">"):
            break
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


def plan_shots(ep: Episode, j: int, K: int, replan_every: int) -> list[Shot]:
    plan = ep.plans[j]
    end = ep.plans[j + 1].step if j + 1 < len(ep.plans) else len(ep.actions)
    out = [Shot(plan.step, j, 0, None, plan.locked, None, "drafting")] * DRAFT_HOLD
    for k in range(1, K + 1):
        status = f"denoise {k:02d}/{K:02d}"
        out += [Shot(plan.step, j, k, k - 1, plan.locked, None, status)] * DENOISE_HOLD
    full = Shot(plan.step, j, K, None, plan.locked, None, f"denoise {K:02d}/{K:02d}")
    out += [full] * ROUTE_HOLD
    for i in range(plan.step, end):
        p = ep.positions[i]
        won = ep.won and i + 1 == len(ep.actions)
        status = "win" if won else f"execute {p - plan.locked + 1:02d}/{replan_every}"
        out += [Shot(i + 1, j, K, None, p, p, status)] * MOVE_HOLD
    return out


def schedule(ep: Episode, K: int, replan_every: int) -> list[Shot]:
    moves, plans = len(ep.actions), len(ep.plans)
    status = f"stairs in {moves} moves"
    poster = Shot(moves, plans - 1, K, None, ep.positions[-1] + 1, None, status)
    motion = [s for j in range(plans) for s in plan_shots(ep, j, K, replan_every)]
    return [poster] * POSTER_HOLD + motion + [poster]


def view_for(ep: Episode, mask: int) -> View:
    box = map_box(ep.states)
    w, h = (box[3] - box[1]) * PITCH, (box[2] - box[0]) * PITCH
    area_w, area_h = MAP_AREA[2] - MAP_AREA[0], MAP_AREA[3] - MAP_AREA[1]
    if w > area_w or h > area_h:
        raise ValueError(f"a {w}x{h} px map does not fit in {area_w}x{area_h}")
    x = MAP_AREA[0] + (area_w - w) // 2
    y = MAP_AREA[1] + (area_h - h) // 2
    stairs = np.argwhere(ep.states[0][1] == ord(">"))[0]
    return View(box, x, y, (int(stairs[0]), int(stairs[1])), mask)


def corner(view: View, cell: tuple[int, int]) -> tuple[int, int]:
    return (
        view.x + (cell[1] - view.box[1]) * PITCH,
        view.y + (cell[0] - view.box[0]) * PITCH,
    )


def centre(view: View, cell: tuple[int, int]) -> tuple[int, int]:
    x, y = corner(view, cell)
    return x + PITCH // 2, y + PITCH // 2


def legend(d: ImageDraw.ImageDraw, x: float, y: int) -> float:
    items = (("masked", 0), ("new", 2), ("drafted", 1), ("now", 3), ("done", 4))
    for i, (label, state) in enumerate(items):
        if i:
            x += 16
        d.rectangle((x, y - 9, x + 9, y), fill=TOKEN_FILLS[state])
        x += 16 + caps(d, x + 16, y, label)
    return x


def map_key(d: ImageDraw.ImageDraw, x: float, y: int) -> None:
    for i, (label, (fill, width)) in enumerate(
        (("route ahead", AHEAD), ("walked", WALKED))
    ):
        if i:
            x += 16
        d.line([(x, y - 5), (x + 15, y - 5)], fill=fill, width=width)
        x += 24 + caps(d, x + 24, y, label)


def base_frame(ep: Episode, view: View) -> Image.Image:
    label = ep.env_id.removeprefix("MiniHack-").removesuffix("-v0")
    img = chrome(
        [("Sixty-four moves, drafted ", False), ("all at once", True)],
        "NeurIPS 2026 · BeNTo",
        f"{label} · eval layout {ep.ep}",
        f"wins {ENV_WIN_RATES[ep.env_id]} here · {ID_WIN_RATE} in-distribution",
    )
    d = ImageDraw.Draw(img)
    y0, x0, y1, x1 = view.box
    w, h = (x1 - x0) * PITCH, (y1 - y0) * PITCH
    corner_ticks(d, (view.x - 4, view.y - 4, view.x + w + 3, view.y + h + 3))
    for x, counter in zip(COUNTERS, ("step", "plans", "result"), strict=True):
        caps(d, x, LABEL_Y, counter)
    right = RIGHT + (GRID_COLS - 1) * TOKEN_PITCH + TOKEN
    top = GRID_ROWS[0]
    corner_ticks(d, (RIGHT - 3, top - 3, right + 2, top + TOKEN + 2))
    if legend(d, RIGHT, LEGEND_Y) > W - M:
        raise ValueError("legend overflows the token grid")
    caps(d, RIGHT, NOTE_Y, "row 1 runs, then the planner replans")
    map_key(d, RIGHT, KEY_Y)
    return img


def polyline(
    d: ImageDraw.ImageDraw, points: list[tuple[int, int]], fill: str, width: int
) -> None:
    if len(points) < 2:
        return
    d.line(points, fill=fill, width=width, joint="curve")


@functools.cache
def ink_offset(text: str) -> tuple[int, int]:
    img = Image.new("L", (2 * PITCH, 2 * PITCH))
    ImageDraw.Draw(img).text(
        (PITCH, PITCH), text, font=font("value", 20), fill=255, anchor="mm"
    )
    left, top, right, bottom = img.getbbox()
    return round((left + right - 1) / 2) - PITCH, round((top + bottom - 1) / 2) - PITCH


def glyph(d: ImageDraw.ImageDraw, view: View, cell: tuple[int, int], ch: str) -> None:
    if ch == " ":
        return
    x, y = centre(view, cell)
    fill = GLYPH_FILLS.get(ch, MUTED)
    if fill == INK:
        d.rectangle((x - 8, y - 9, x + 7, y + 8), fill=BG)
    text = "·" if ch == "." else ch
    dx, dy = ink_offset(text)
    d.text((x - dx, y - dy), text, font=font("value", 20), fill=fill, anchor="mm")


def ahead(ep: Episode, shot: Shot, mask: int) -> list[tuple[int, int]]:
    if shot.t == len(ep.actions):
        return []
    chars = ep.states[shot.t][1]
    start = shot.done if shot.now is None else shot.now + 1
    tokens = ep.plans[shot.plan].rows[shot.k][start:]
    return route(chars, agent_cell(chars), tokens, mask)


def draw_map(d: ImageDraw.ImageDraw, ep: Episode, shot: Shot, view: View) -> None:
    chars = ep.states[shot.t][1]
    here = agent_cell(chars)
    trail = [centre(view, agent_cell(c)) for _, c in ep.states[: shot.t + 1]]
    polyline(d, trail, *WALKED)
    route_ahead = [here, *ahead(ep, shot, view.mask)]
    polyline(d, [centre(view, c) for c in route_ahead], *AHEAD)
    y0, x0, y1, x1 = view.box
    for (y, x), c in np.ndenumerate(chars[y0:y1, x0:x1]):
        glyph(d, view, (y0 + y, x0 + x), chr(c))
    x, y = corner(view, view.stairs)
    d.rectangle((x, y, x + PITCH - 1, y + PITCH - 1), outline=INK)


@functools.cache
def token_cell(token: int, state: int) -> Image.Image:
    scale = 4
    img = Image.new("RGB", (scale * TOKEN, scale * TOKEN), TOKEN_FILLS[state])
    if state == 0:
        return img.reduce(scale)
    dy, dx = MOVES[token % len(MOVES)]
    angle = math.atan2(dy, dx)
    cos, sin = math.cos(angle), math.sin(angle)
    mid = scale * TOKEN / 2
    points = [
        (mid + scale * (x * cos - y * sin), mid + scale * (x * sin + y * cos))
        for x, y in ARROW
    ]
    ImageDraw.Draw(img).polygon(points, fill=ARROW_FILLS[state])
    return img.reduce(scale)


def token_grid(img: Image.Image, row: np.ndarray, states: np.ndarray) -> None:
    for i, (token, state) in enumerate(zip(row.tolist(), states.tolist(), strict=True)):
        x = RIGHT + i % GRID_COLS * TOKEN_PITCH
        img.paste(token_cell(token, state), (x, GRID_ROWS[i // GRID_COLS]))


def header(d: ImageDraw.ImageDraw, seq_len: int, status: str) -> None:
    used = caps(d, RIGHT, HEAD_Y, f"plan · {seq_len} moves")
    used += caps(d, W - M, HEAD_Y, status, align="right", fill=INK)
    if used > W - 2 * M - RIGHT:
        raise ValueError(f"plan header is {used:.0f} px, over {W - 2 * M - RIGHT}")


def draw_counters(d: ImageDraw.ImageDraw, ep: Episode, shot: Shot) -> None:
    won = ep.won and shot.t == len(ep.actions)
    value(d, COUNTERS[0], VALUE_Y, f"{shot.t:03d}")
    value(d, COUNTERS[1], VALUE_Y, f"{shot.plan + 1}")
    value(d, COUNTERS[2], VALUE_Y, "WIN" if won else "–", fill=INK if won else MUTED)


def draw_frame(base: Image.Image, ep: Episode, shot: Shot, view: View) -> Image.Image:
    plan = ep.plans[shot.plan]
    row = plan.rows[shot.k]
    prev = None if shot.prev_k is None else plan.rows[shot.prev_k]
    states = token_states(row, prev, shot.done, shot.now, view.mask)
    img = base.copy()
    d = ImageDraw.Draw(img)
    draw_map(d, ep, shot, view)
    draw_counters(d, ep, shot)
    header(d, len(row), shot.status)
    token_grid(img, row, states)
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
    if (cfg.seq_len, cfg.replan_every) != (len(GRID_ROWS) * GRID_COLS, GRID_COLS):
        raise ValueError("the token grid draws 64 moves, 16 per plan window")
    # NLE blanks the observation once the agent takes the stairs.
    ep = replace(ep, states=[*ep.states[:-1], arrive(ep.states[:-1])])
    view = view_for(ep, cfg.mask_token)
    base = base_frame(ep, view)
    K = len(ep.plans[0].rows) - 1
    shots = schedule(ep, K, cfg.replan_every)
    return [draw_frame(base, ep, shot, view) for shot in shots]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--env", choices=list(ENV_WIN_RATES), default=next(iter(ENV_WIN_RATES))
    )
    p.add_argument(
        "--episode", type=int, help="render this layout, skipping the search"
    )
    p.add_argument("--episodes", type=int, default=50, help="layouts searched")
    p.add_argument("--min-steps", type=int, default=4, help="skip shorter wins")
    p.add_argument(
        "--max-steps", type=int, help="default: replan_every, so one plan wins"
    )
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
    max_steps = args.max_steps or cfg.replan_every
    run = first_win(model, cfg, args.env, episodes, args.min_steps, max_steps)
    print(
        f"selected ep {run.ep}: first of {episodes.start}..{episodes.stop - 1} won "
        f"in {args.min_steps}-{max_steps} moves ({len(run.actions)} moves, "
        f"plans: {len(run.plans)})"
    )
    frames = render(run, cfg)
    size = encode_gif(frames, args.out, COLOURS, MAX_BYTES)
    print(f"{args.out}: {len(frames)} frames, {size:,} B")


if __name__ == "__main__":
    main()
