"""
Utility functions — visualization, inference, logging, and artifact management.
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from config import HYPERPARAMS, ACTION_DIM, MASK_TOKEN, PAD_TOKEN, ACTION_NAMES

HF_ARCHIVE_REPO = "remdm-minihack/ReMDM-MiniHack"  # Load checkpoints, datasets
HF_GENERIZATION_REPO = "remdm-minihack/ReMDM-MiniHack"  # Main repo with generalization folder

# ---------------------------------------------------------------------------
# Staircase detection (for aux goal loss in Trainer)
# ---------------------------------------------------------------------------
_STAIR_GLYPHS = (ord(">"), 2368, 2310, 2383)


def find_staircase_from_glyphs(glyphs):
    """Return normalized (row, col) of staircase in glyph map, or (-1, -1)."""
    if glyphs is None or not hasattr(glyphs, "shape"):
        return (-1.0, -1.0)
    for gid in _STAIR_GLYPHS:
        pos = np.argwhere(glyphs == gid)
        if len(pos) > 0:
            r, c = pos[0]
            h, w = glyphs.shape
            return (r / max(1, h - 1), c / max(1, w - 1))
    return (-1.0, -1.0)


# ---------------------------------------------------------------------------
# Artifact helpers: Hugging Face Hub
# ---------------------------------------------------------------------------

def load_checkpoint(version, filename="inference_weights.pth", device=None, token=None):
    """Load a checkpoint from Hugging Face Hub. Pass token for private repos."""
    dev = device or HYPERPARAMS["device"]
    from huggingface_hub import hf_hub_download
    token = token or _get_hf_token()
    repo_path = f"{version}/{filename}"
    local = hf_hub_download(repo_id=HF_ARCHIVE_REPO, filename=repo_path, token=token)
    return torch.load(local, map_location=dev, weights_only=False)


def load_dataset(version, filename=None, token=None):
    """Load an oracle dataset from Hugging Face Hub. Pass token for private repos."""
    fname = filename or f"oracle_demos_{version}.pt"
    from huggingface_hub import hf_hub_download
    token = token or _get_hf_token()
    local = hf_hub_download(repo_id=HF_ARCHIVE_REPO, filename=f"datasets/{fname}", token=token)
    return torch.load(local, weights_only=False)


def load_dataset_metadata(version, token=None):
    """Load collection metadata JSON from Hugging Face Hub. Pass token for private repos."""
    fname = f"oracle_demos_{version}_meta.json"
    try:
        from huggingface_hub import hf_hub_download
        token = token or _get_hf_token()
        local = hf_hub_download(repo_id=HF_ARCHIVE_REPO, filename=f"datasets/{fname}", token=token)
        with open(local) as f:
            return json.load(f)
    except Exception:
        return None


def _get_hf_token():
    """Get HF token from Colab secrets or env."""
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except (ImportError, Exception):
        pass
    return os.environ.get("HF_TOKEN")


def upload_generalization_to_hub(local_dir, run_id, token=None):
    """Upload all artifacts from local_dir to HF remdm-minihack/ReMDM-MiniHack/generalization folder."""
    from huggingface_hub import HfApi
    import os
    
    token = token or _get_hf_token()
    api = HfApi(token=token)
    
    # Path within the repo: generalization/run_id/file (e.g. generalization/20250218_1430/generalization_final.pth)
    repo_path_base = f"generalization/{run_id}"

    def _upload(local_path, repo_path):
        if os.path.exists(local_path):
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=HF_GENERIZATION_REPO,
                token=token,
            )
            print(f"  Uploaded: {repo_path}")

    print(f"Pushing {run_id}/ to {HF_GENERIZATION_REPO}/generalization/...")
    for fname in os.listdir(local_dir):
        full = os.path.join(local_dir, fname)
        if os.path.isfile(full):
            _upload(full, f"{repo_path_base}/{fname}")
    print(f"Done. View at: https://huggingface.co/{HF_GENERIZATION_REPO}/tree/main/generalization/{run_id}")


def _unpack_logits(output):
    """Handle dict (actions), tuple (logits, goal_pred), or raw tensor."""
    if isinstance(output, dict):
        return output["actions"]
    return output[0] if isinstance(output, tuple) else output


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_crop_grid(env, n=4, seed=42):
    """Display a grid of local 9x9 crops from random environments."""
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    for i in range(n):
        obs, _ = env.reset(seed=seed + i)
        local_crop = obs[0]
        axes[i].imshow(local_crop, cmap='terrain')
        axes[i].set_title(env.current_env_id.split('-')[1], fontsize=9)
        axes[i].axis('off')
    plt.suptitle("Local Crops (9x9)")
    plt.tight_layout()
    plt.show()


def plot_remdm_analytics(tracking_confidence, tracking_masked_count):
    """
    Dual-axis plot of the ReMDM diffusion loop analytics.
    Left y-axis: average confidence — commitment rising.
    Right y-axis: masked token count — uncertainty decreasing.
    """
    steps = list(range(1, len(tracking_confidence) + 1))

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_conf = '#2196F3'
    ax1.set_xlabel('Diffusion Step', fontsize=12)
    ax1.set_ylabel('Avg Confidence (unmasked tokens)', color=color_conf, fontsize=12)
    ax1.plot(steps, tracking_confidence, 'o-', color=color_conf,
             linewidth=2.5, markersize=8, label='Avg Confidence')
    ax1.tick_params(axis='y', labelcolor=color_conf)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    color_mask = '#FF7043'
    ax2.set_ylabel('Masked Token Count', color=color_mask, fontsize=12)
    ax2.bar(steps, tracking_masked_count, alpha=0.35, color=color_mask,
            width=0.6, label='Masked Tokens')
    ax2.tick_params(axis='y', labelcolor=color_mask)
    ax2.set_ylim(0, max(tracking_masked_count) * 1.2 if tracking_masked_count else 1)

    ax1.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)

    plt.title('ReMDM Diffusion Loop \u2014 "Thinking" Process', fontsize=14, fontweight='bold')
    ax1.set_xticks(steps)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _check_hazard(local_crop, action):
    """
    Check if an action from the agent's center tile steps into a hazard.
    local_crop is a (crop_size, crop_size) array of NetHack glyph IDs.
    Actions: 0=N, 1=E, 2=S, 3=W.
    Returns True if the target tile is a wall, locked door, lava, or water.
    """
    crop_size = HYPERPARAMS["crop_size"]
    cy, cx = crop_size // 2, crop_size // 2
    dy, dx = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}.get(action, (0, 0))
    ny, nx = cy + dy, cx + dx
    if not (0 <= ny < crop_size and 0 <= nx < crop_size):
        return True
    glyph = int(local_crop[ny, nx])
    # NetHack CMAP glyph IDs
    HAZARD_GLYPHS = {2359, 2360, 2389, 2390}  # wall-v, wall-h, water, lava
    # Char-encoded fallbacks (some envs use raw chars as glyphs)
    HAZARD_CHARS = {ord('|'), ord('-'), ord('+'), ord('L'), ord('W')}
    return glyph in HAZARD_GLYPHS or glyph in HAZARD_CHARS


def run_inference_test(model, local_obs, global_obs, diffusion_steps=5,
                       valid_actions=8, physics_aware=True):
    """
    ReMDM inference loop: all-MASK → progressively denoised action path.

    Three safety nets prevent degenerate behaviour on untrained models:

    A. **Minimum keep (10%)**: At every step, at least 10% of tokens are
       unmasked so the model always has partial context to build on.

    B. **Physics softener**: Hazardous actions get confidence=0.001 (not 0.0),
       so they rank lowest but still survive if all alternatives are worse.

    C. **Final-step force**: On the last step, mask_condition is forced False
       — the model commits to a full 64-token trajectory with no MASK left.

    Returns (path_per_step, final_seq, tracking_confidence, tracking_masked_count).
    """
    N_PHYSICS_CHECK = 8

    dev = HYPERPARAMS["device"]
    local_t = torch.tensor(local_obs, dtype=torch.long).unsqueeze(0).to(dev)
    global_t = torch.tensor(global_obs, dtype=torch.long).unsqueeze(0).to(dev)
    seq_len = HYPERPARAMS["seq_len"]
    seq = torch.full((1, seq_len), MASK_TOKEN, dtype=torch.long, device=dev)

    model.eval()
    path_per_step = []
    tracking_confidence = []
    tracking_masked_count = []

    with torch.no_grad():
        for step in range(1, diffusion_steps + 1):
            ratio = step / diffusion_steps
            t_val = int(100 * (1.0 - ratio))
            t_tensor = torch.tensor([t_val], dtype=torch.long, device=dev)
            logits = _unpack_logits(model(local_t, global_t, seq, t_tensor))
            if valid_actions < ACTION_DIM:
                logits[0, :, valid_actions:] = -float('inf')

            #########################
            ### As flagged by Ali ###
            #########################
            # 1. APPLY TEMPERATURE AND TOP-K FILTERING
            temperature = 0.5 # T<1.0 makes it greedy but fluid
            top_k = 4 # Only sample from the top 4 ideas

            scaled_logits = logits/temperature

            # Apply Top-K Filtering
            values,indices = torch.topk(scaled_logits, top_k, dim=1)
            filtered_logits = torch.full_like(scaled_logits, -float('inf'))
            filtered_logits.scatter_(-1, indices, values)

            # Get probabilities from filtered distribution
            probs = torch.softmax(filtered_logits, dim=-1)

            # Sample predictions instead of taking the rigid argmax
            action_dist = torch.distributions.Categorical(probs)
            preds = action_dist.sample() # Shape: (Batch, Seq_Len)

            # Get the confidence of the SPECIFIC actions we just sampled
            confidences, preds = torch.gather(probs, -1, preds.unsqueeze(-1)).squeeze(-1)

            ### END ###

            # 2. Physics-aware hazard penalty (Safety Net B: soft, not zero)
            if physics_aware:
                preds_np = preds[0].cpu().numpy()
                conf_override = confidences.clone()
                for pos_idx in range(min(N_PHYSICS_CHECK, seq_len)):
                    pred_action = int(preds_np[pos_idx])
                    if pred_action <= 3 and _check_hazard(local_obs, pred_action):
                        conf_override[0, pos_idx] = 0.001
                confidences = conf_override

            # 3. Schedule thresholding (Safety Net A: always keep >= 10%)
            min_keep = int(seq_len * 0.10)
            schedule_keep = int(seq_len * ratio)
            num_to_unmask = max(min_keep, schedule_keep)
            sorted_conf, _ = torch.sort(confidences[0], descending=True)
            dynamic_threshold = sorted_conf[num_to_unmask - 1].item()
            mask_condition = confidences[0] < dynamic_threshold

            # 4. Safety Net C: final step — force full commitment
            if step == diffusion_steps:
                mask_condition[:] = False

            # 5. Apply mask & analytics
            seq = preds.clone()
            seq[0, mask_condition] = MASK_TOKEN

            unmasked_confs = confidences[0][~mask_condition]
            avg_conf = unmasked_confs.mean().item() if unmasked_confs.numel() > 0 else 0.0
            tracking_confidence.append(avg_conf)
            tracking_masked_count.append(int(mask_condition.sum().item()))

            path_per_step.append(seq[0].cpu().numpy().copy())

    final_seq = seq[0].cpu().numpy()
    return path_per_step, final_seq, tracking_confidence, tracking_masked_count


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def save_run_log(out_dir, tracking_conf=None, tracking_masked=None,
                 final_seq=None, diffusion_steps=5, situation=None):
    """Write a timestamped run log to out_dir."""
    lines = [
        "ReMDM-MiniHack-Project \u2014 Run Log",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Device: {HYPERPARAMS['device']}",
        f"Seq len: {HYPERPARAMS['seq_len']} | Crop: {HYPERPARAMS['crop_size']}",
    ]
    if situation:
        lines.append(f"Situation: {situation}")
    lines.append("")

    if tracking_conf is not None and tracking_masked is not None:
        lines.append("--- Module 7: ReMDM Analytics ---")
        lines.append(f"Diffusion steps: {diffusion_steps}")
        for i, (c, m) in enumerate(zip(tracking_conf, tracking_masked)):
            lines.append(f"  Step {i+1}: confidence={c:.4f}, masked={m}")
        lines.append("")

    if final_seq is not None:
        readable = [ACTION_NAMES[a] if a < len(ACTION_NAMES) else '?' for a in final_seq[:16]]
        lines.append(f"Generated path (first 16): {readable}")
        lines.append(f"Unmasked tokens: {np.sum(final_seq != MASK_TOKEN)}/{HYPERPARAMS['seq_len']}")
        lines.append("")

    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(out_dir, f"run_{timestamp}.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Run log saved to: {log_path}")
    return log_path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def select_action(model, local_obs, global_obs, diffusion_steps=5,
                  valid_actions=ACTION_DIM, blind_global=False):
    """Generate one action using the diffusion planner (greedy first-token).

    EDIT: greedy first-token --> stochastic first-token with Temp/Top-K

    blind_global: if True, zero out global obs (ablation: local-only).
    """
    dev = HYPERPARAMS["device"]
    local_t = torch.tensor(local_obs, dtype=torch.long).unsqueeze(0).to(dev)
    if blind_global:
        global_obs = np.zeros_like(global_obs)
    global_t = torch.tensor(global_obs, dtype=torch.long).unsqueeze(0).to(dev)
    seq_len = HYPERPARAMS["seq_len"]
    seq = torch.full((1, seq_len), MASK_TOKEN, dtype=torch.long, device=dev)

    with torch.no_grad():
        #logits = None
        for step in range(1, diffusion_steps + 1):
            ratio = step / diffusion_steps
            t_val = int(100 * (1.0 - ratio))
            t_tensor = torch.tensor([t_val], dtype=torch.long, device=dev)
            logits = _unpack_logits(model(local_t, global_t, seq, t_tensor))
            if valid_actions < ACTION_DIM:
                logits[0, :, valid_actions:] = -float('inf')
            
            #########################
            ### As flagged by Ali ###
            #########################

            # 1. Apply Temperature and Top-K Filtering
            temperature = 0.5
            top_k = 4

            scaled_logits = logits / temperature
            values, indices = torch.topk(scaled_logits, top_k, dim=-1)
            filtered_logits = torch.full_like(scaled_logits,-float('inf'))
            filtered_logits.scatter_(-1,indices,values)

            probs = torch.softmax(filtered_logits, dim=-1)
            action_dist = torch.distributions.Categorical(probs)
            preds = action_dist.sample()

            confidences = torch.gather(probs,-1,preds.unsqueeze(-1)).squeeze(-1)

            # 2. Dynamic Masking (Standard MaskGIT)
            num_to_unmask = max(1, int(seq_len * ratio))
            sorted_conf, _ = torch.sort(confidences[0], descending=True)
            threshold = sorted_conf[num_to_unmask - 1]
            seq = preds.clone()
            seq[confidences < threshold] = MASK_TOKEN

            # 3. ReMDM Stochastic Remasking
            # Randomly re-mask a fraction of the KEPT tokens to allow revision
            if step < diffusion_steps:
                remask_prob = 0.15*(1.0-ratio)
                is_unmasked = (seq != MASK_TOKEN)
                remdm_mask = (torch.rand(seq.shape,device=dev) < remask_prob)& is_unmasked
                seq[remdm_mask] = MASK_TOKEN
        # Extract the Final Action        
        action = seq[0, 0].item()

        # Safety Fallback
        if action >= valid_actions:
            action = 0
    return action


def run_episode(model, env_id=None, des_content=None, diffusion_steps=5,
                max_steps=200, seed=None, record_maps=False, blind_global=False):
    """Run one episode with the diffusion planner.

    Provide either env_id (MiniHack registry name) or des_content (.des string).
    Set record_maps=True to capture char-maps for trajectory visualisation.
    blind_global: if True, zero out global obs (ablation).
    """
    from env import AdvancedObservationEnv

    if des_content:
        env = AdvancedObservationEnv([], des_files=[("custom", des_content)])
    else:
        env = AdvancedObservationEnv([env_id])

    obs, _ = env.reset(seed=seed)
    model.eval()
    valid_actions = env.current_env.action_space.n

    total_reward = 0.0
    steps = 0
    won = False
    maps, actions_taken, rewards_list = [], [], []

    for _ in range(max_steps):
        if record_maps and env.last_raw_obs is not None:
            maps.append(env.last_raw_obs['chars'].copy())

        local_obs, global_obs = obs
        action = select_action(model, local_obs, global_obs,
                               diffusion_steps, valid_actions, blind_global=blind_global)
        try:
            obs, reward, term, trunc, info = env.step(action)
        except RuntimeError as e:
            if "finished NetHack" in str(e):
                break
            raise

        total_reward += reward
        steps += 1
        if record_maps:
            actions_taken.append(action)
            rewards_list.append(reward)

        if info.get('won', False):
            won = True
        if term or trunc:
            break

    if record_maps and env.last_raw_obs is not None:
        maps.append(env.last_raw_obs['chars'].copy())

    env.current_env.close()

    result = {
        'env_id': env_id or 'custom',
        'seed': seed,
        'won': won,
        'total_reward': total_reward,
        'steps': steps,
    }
    if record_maps:
        result['maps'] = maps
        result['actions'] = actions_taken
        result['rewards'] = rewards_list
    return result


def run_model_episode(model, env, diffusion_steps=5, replan_every=16,
                      max_steps=200, device='cuda', seed=None):
    """Roll out the diffusion planner in a live environment.

    Always calls env.reset() internally at the start. Caller must NOT reset
    before calling. This ensures self-containment and correct env_id capture.

    Args:
        model: LocalDiffusionPlannerWithGlobal (or compatible). Must be in eval mode.
        env: AdvancedObservationEnv instance. Will be reset by this function.
        diffusion_steps: Number of denoising steps to generate action sequence.
        replan_every: Execute this many actions from the plan before replanning.
        max_steps: Maximum total environment steps before giving up.
        device: Torch device.
        seed: Optional seed for env.reset (for reproducibility / DAgger re-labeling).

    Returns:
        dict: 'won' (bool), 'states' (list of (local, global) tuples),
              'actions' (list of int), 'total_reward' (float), 'steps' (int),
              'env_id' (str from env.current_env_id after reset).
    """
    seq_len = HYPERPARAMS["seq_len"]
    obs, info = env.reset(seed=seed)
    env_id = env.current_env_id

    states, actions = [], []
    total_reward = 0.0
    step_count = 0
    plan = None
    plan_idx = 0
    won = False

    while step_count < max_steps:
        if plan is None or plan_idx >= replan_every:
            local_crop, global_map = obs
            obs_local_t = torch.tensor(local_crop, dtype=torch.long).unsqueeze(0).to(device)
            obs_global_t = torch.tensor(global_map, dtype=torch.long).unsqueeze(0).to(device)
            seq = torch.full((1, seq_len), MASK_TOKEN, dtype=torch.long, device=device)

            with torch.no_grad():
                for step in range(1, diffusion_steps + 1):
                    ratio = step / diffusion_steps
                    t_val = int(100 * (1.0 - ratio))
                    t_tensor = torch.tensor([t_val], dtype=torch.long, device=device)
                    output = model(obs_local_t, obs_global_t, seq, t_tensor)
                    logits = output["actions"] if isinstance(output, dict) else (output[0] if isinstance(output, tuple) else output)
                    if env.valid_action_count < ACTION_DIM:
                        logits[0, :, env.valid_action_count:] = -float('inf')
                    probs = torch.softmax(logits, dim=-1)
                    confidences, preds = torch.max(probs, dim=-1)
                    num_to_unmask = max(1, int(seq_len * ratio))
                    sorted_conf, _ = torch.sort(confidences[0], descending=True)
                    threshold = sorted_conf[num_to_unmask - 1]
                    seq = preds.clone()
                    seq[confidences < threshold] = MASK_TOKEN

            plan = seq[0].cpu().numpy()
            plan_idx = 0

        action = int(plan[plan_idx])
        plan_idx += 1

        states.append(obs)
        actions.append(action)

        try:
            obs, reward, term, trunc, info = env.step(action)
        except RuntimeError as e:
            if "finished NetHack" in str(e):
                break
            raise

        total_reward += reward
        step_count += 1
        if info.get('won', False):
            won = True
        if term or trunc:
            break

    return {
        'won': won,
        'states': states,
        'actions': actions,
        'total_reward': total_reward,
        'steps': step_count,
        'env_id': env_id,
    }


def benchmark_inference(model, n_actions=100, diffusion_steps=5):
    """Measure inference speed. Returns (diffusion_steps_per_sec, actions_per_sec)."""
    import time
    model.eval()
    local_dummy = np.zeros((9, 9), dtype=np.int16)
    global_dummy = np.zeros((21, 79), dtype=np.int16)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_actions):
        select_action(model, local_dummy, global_dummy, diffusion_steps)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed = t1 - t0
    total_steps = n_actions * diffusion_steps
    steps_per_sec = total_steps / elapsed if elapsed > 0 else 0
    actions_per_sec = n_actions / elapsed if elapsed > 0 else 0
    return steps_per_sec, actions_per_sec


def evaluate_envs(model, env_ids=None, des_files=None, episodes=50,
                  diffusion_steps=5, base_seed=42, max_steps=200, blind_global=False):
    """Evaluate model across multiple environments.

    env_ids:   list of MiniHack registry IDs
    des_files: list of .des file paths
    blind_global: if True, zero out global obs (ablation: local-only).

    Returns a list of per-environment result dicts.
    """
    configs = []
    for eid in (env_ids or []):
        configs.append((eid, None))
    for des_path in (des_files or []):
        name = os.path.basename(des_path).replace('.des', '')
        with open(des_path) as f:
            configs.append((name, f.read()))

    print(f"\n{'Environment':<40} | {'Eps':>4} | "
          f"{'Wins':>6} | {'Win%':>6} | {'Reward':>7} | {'Steps':>6}")
    print("-" * 85)

    all_results = []
    for env_id, des_content in configs:
        env_results = []
        for ep in range(episodes):
            seed = base_seed + ep if base_seed is not None else None
            r = run_episode(model,
                            env_id=env_id if not des_content else None,
                            des_content=des_content,
                            diffusion_steps=diffusion_steps,
                            max_steps=max_steps, seed=seed, blind_global=blind_global)
            env_results.append(r)

        wins = sum(1 for r in env_results if r['won'])
        avg_rew = np.mean([r['total_reward'] for r in env_results])
        avg_steps = np.mean([r['steps'] for r in env_results])

        all_results.append({
            'env_id': env_id,
            'episodes': episodes,
            'wins': wins,
            'win_rate': wins / episodes if episodes > 0 else 0,
            'avg_reward': avg_rew,
            'avg_steps': avg_steps,
            'details': env_results,
        })

        print(f"  {env_id:<40} | {episodes:>4} | "
              f"{wins:>6} | {wins / episodes * 100:>5.1f}% | "
              f"{avg_rew:>7.1f} | {avg_steps:>6.1f}")

    return all_results


def plot_eval_results(results):
    """Bar chart of win rates across environments."""
    names = [r['env_id'] for r in results]
    rates = [r['win_rate'] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))
    bars = ax.bar(range(len(names)), rates, color='#2196F3', alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('ReMDM Evaluation \u2014 Win Rate by Environment')
    ax.set_ylim(0, 105)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def render_trajectory(episode_data):
    """Visualise episode trajectory: ASCII overview + matplotlib plot."""
    if 'maps' not in episode_data or not episode_data['maps']:
        print("No map data recorded (run with record_maps=True).")
        return

    maps = episode_data['maps']
    actions = episode_data.get('actions', [])

    def _map_str(char_map):
        lines = []
        for row in char_map:
            line = ''.join(chr(c) if 32 <= c < 127 else ' ' for c in row)
            if line.strip():
                lines.append(line)
        return '\n'.join(lines)

    positions = []
    for char_map in maps:
        rows, cols = np.where(char_map == ord('@'))
        if len(rows) > 0:
            positions.append((int(rows[0]), int(cols[0])))

    status = 'WON' if episode_data['won'] else 'LOST'
    print(f"\n{'=' * 50}")
    print(f"  {episode_data.get('env_id', 'Episode')} \u2014 {status}")
    print(f"  Steps: {episode_data['steps']} | "
          f"Reward: {episode_data['total_reward']:.2f}")
    print(f"{'=' * 50}")

    print("\n--- Initial Map ---")
    print(_map_str(maps[0]))

    if positions:
        overlay = maps[0].copy()
        for i, (r, c) in enumerate(positions):
            if i == 0:
                overlay[r, c] = ord('S')
            elif i == len(positions) - 1:
                overlay[r, c] = ord('W') if episode_data['won'] else ord('X')
            else:
                if overlay[r, c] in (ord('.'), ord('@')):
                    overlay[r, c] = ord('*')
        tag = 'W=win' if episode_data['won'] else 'X=end'
        print(f"\n--- Trajectory (S=start *=path {tag}) ---")
        print(_map_str(overlay))

    if actions:
        names = [ACTION_NAMES[a] if a < len(ACTION_NAMES) else '?'
                 for a in actions]
        print(f"\nActions: {' '.join(names[:40])}")
        if len(actions) > 40:
            print(f"  ... ({len(actions) - 40} more)")

    if not positions:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: map + path
    ax = axes[0]
    char_map = maps[0]
    h, w = char_map.shape
    grid = np.ones((h, w, 3)) * 0.95
    for r in range(h):
        for c in range(w):
            ch = chr(char_map[r, c]) if 32 <= char_map[r, c] < 127 else ' '
            if ch in '|-+':
                grid[r, c] = [0.3, 0.3, 0.3]
            elif ch == '.':
                grid[r, c] = [0.93, 0.91, 0.82]
            elif ch == 'L':
                grid[r, c] = [1.0, 0.3, 0.0]
            elif ch == '>':
                grid[r, c] = [0.0, 0.8, 0.0]

    ax.imshow(grid, aspect='auto')
    xs = [p[1] for p in positions]
    ys = [p[0] for p in positions]
    ax.plot(xs, ys, 'b-', alpha=0.4, linewidth=1.5)
    ax.plot(xs[0], ys[0], 'go', markersize=10, label='Start')
    if episode_data['won']:
        ax.plot(xs[-1], ys[-1], 'g*', markersize=14, label='Win')
    else:
        ax.plot(xs[-1], ys[-1], 'rx', markersize=12,
                markeredgewidth=2, label='Death/End')
    ax.legend(fontsize=9)
    ax.set_title(episode_data.get('env_id', 'Episode'))

    # Right: cumulative reward
    ax2 = axes[1]
    if 'rewards' in episode_data and episode_data['rewards']:
        cum_rew = np.cumsum(episode_data['rewards'])
        ax2.plot(cum_rew, 'g-', alpha=0.7, linewidth=2)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Cumulative Reward')
        ax2.set_title(f"Total Reward: {episode_data['total_reward']:.2f}")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_sample_trajectories(model, env_ids, n_samples=3, diffusion_steps=5,
                             max_steps=200, base_seed=100):
    """Run n_samples episodes per environment and plot agent paths side by side.

    Each environment gets one row of subplots (one per sample).
    """
    import random

    _WALL_CHARS = set(ord(c) for c in '|-+')
    _FLOOR = ord('.')
    _LAVA = ord('L')
    _STAIR = ord('>')
    _AGENT = ord('@')

    def _render_grid(char_map):
        h, w = char_map.shape
        grid = np.ones((h, w, 3)) * 0.95
        for r in range(h):
            for c in range(w):
                v = char_map[r, c]
                if v in _WALL_CHARS:
                    grid[r, c] = [0.3, 0.3, 0.3]
                elif v == _FLOOR:
                    grid[r, c] = [0.93, 0.91, 0.82]
                elif v == _LAVA:
                    grid[r, c] = [1.0, 0.3, 0.0]
                elif v == _STAIR:
                    grid[r, c] = [0.0, 0.8, 0.0]
        return grid

    nrows = len(env_ids)
    fig, axes = plt.subplots(nrows, n_samples, figsize=(5 * n_samples, 4 * nrows),
                             squeeze=False)

    for row, env_id in enumerate(env_ids):
        seeds = random.sample(range(1, 10000), n_samples)
        for col, seed in enumerate(seeds):
            ax = axes[row][col]
            ep = run_episode(model, env_id=env_id, diffusion_steps=diffusion_steps,
                             max_steps=max_steps, seed=seed, record_maps=True)

            positions = []
            for char_map in ep.get('maps', []):
                rs, cs = np.where(char_map == _AGENT)
                if len(rs) > 0:
                    positions.append((int(rs[0]), int(cs[0])))

            if ep.get('maps'):
                grid = _render_grid(ep['maps'][0])
                ax.imshow(grid, aspect='auto')

            if positions:
                xs = [p[1] for p in positions]
                ys = [p[0] for p in positions]
                ax.plot(xs, ys, 'b-', alpha=0.5, linewidth=1.5)
                ax.plot(xs[0], ys[0], 'go', markersize=8)
                if ep['won']:
                    ax.plot(xs[-1], ys[-1], 'g*', markersize=12)
                else:
                    ax.plot(xs[-1], ys[-1], 'rx', markersize=10, markeredgewidth=2)

            status = 'WIN' if ep['won'] else 'LOSS'
            ax.set_title(f"{status} | {ep['steps']} steps | R={ep['total_reward']:.1f}",
                         fontsize=9)
            if col == 0:
                ax.set_ylabel(env_id.replace('MiniHack-', ''), fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Sample Trajectories (green=start, star=win, x=loss)", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.show()
