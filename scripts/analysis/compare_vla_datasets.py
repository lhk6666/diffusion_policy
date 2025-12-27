#!/usr/bin/env python3
"""Compare VLA Zarr datasets for Diffusion Policy training.

This script is intentionally lightweight (pure Python + numpy/zarr) and prints
metrics that correlate strongly with DP diffusion loss behavior:
- episode length distribution
- effective repetition introduced by clipping/padding for a given horizon
- action dynamics (smoothness / step size)
- instruction text diversity/length

Examples:
  python scripts/analysis/compare_vla_datasets.py \
    --a /media/dragon_llm/linux_ssd/vla_dp_224/train \
    --b /media/dragon_llm/linux_ssd/vla_dataset_unified_static_v10/train \
    --horizon 128 --pad-before 0 --pad-after 0 --sample-mode episode

Notes:
- For split directories, the zarr store may be either <split>/dataset.zarr or
  directly at <split>/ (zarr directory store). Both are supported.
- episode_meta.json may live alongside the zarr store.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import zarr


def _resolve_zarr_dir(split_dir: Path) -> Path:
    """Return the directory that should be opened by zarr.open()."""
    if split_dir.is_dir():
        nested = split_dir / "dataset.zarr"
        if nested.exists() and nested.is_dir():
            return nested
        return split_dir
    return split_dir


def _find_episode_meta(zarr_dir: Path) -> Optional[Path]:
    candidates = [zarr_dir / "episode_meta.json", zarr_dir.parent / "episode_meta.json"]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def _percentiles(x: np.ndarray, ps: Iterable[int]) -> Dict[int, float]:
    return {int(p): float(np.percentile(x, p)) for p in ps}


@dataclass
class DatasetSummary:
    name: str
    zarr_dir: Path
    n_steps: int
    n_episodes: int
    n_images: int
    episode_len_min: int
    episode_len_mean: float
    episode_len_max: int
    episode_len_percentiles: Dict[int, float]
    img_dtype: str
    img_min: float
    img_max: float
    action_min: float
    action_max: float
    action_mean: float
    action_std: float
    instruction_count: int
    instruction_unique: int
    instruction_len_mean: float
    instruction_len_p95: float
    repetition_frac_mean: float
    repetition_frac_p95: float
    repeated_tail_len_mean: float
    step_delta_l2_mean: float
    step_delta_l2_p95: float
    action_minus_pos_l2_mean: float
    action_minus_pos_l2_p95: float


def summarize_dataset(
    name: str,
    split_path: str,
    horizon: int,
    pad_before: int,
    pad_after: int,
    sample_mode: str,
    max_episodes_for_dynamics: int,
    seed: int,
) -> DatasetSummary:
    split_dir = Path(split_path)
    zarr_dir = _resolve_zarr_dir(split_dir)
    root = zarr.open(str(zarr_dir), mode="r")

    # required arrays
    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    episode_starts = np.concatenate([[0], episode_ends[:-1]]).astype(np.int64)
    lens = (episode_ends - episode_starts).astype(np.int64)

    states = np.asarray(root["data/state"][:], dtype=np.float32)
    actions = np.asarray(root["data/action"][:], dtype=np.float32)
    agent_pos = np.asarray(root["data/agent_pos"][:], dtype=np.float32)
    images = root["data/img"]

    # image range sample
    img_sample = np.asarray(images[: min(64, images.shape[0])])
    img_min = float(img_sample.min())
    img_max = float(img_sample.max())

    # action sample
    act_sample = actions[: min(200_000, actions.shape[0])]

    # instruction stats
    meta_path = _find_episode_meta(zarr_dir)
    instructions: List[str] = []
    if meta_path is not None:
        meta = json.loads(meta_path.read_text())
        if isinstance(meta, list):
            for item in meta:
                if isinstance(item, dict):
                    instr = item.get("instruction")
                    if isinstance(instr, str):
                        instructions.append(instr)
        elif isinstance(meta, dict):
            # uncommon format: dict keyed by episode id
            for _, item in meta.items():
                if isinstance(item, dict):
                    instr = item.get("instruction")
                    if isinstance(instr, str):
                        instructions.append(instr)

    instr_lens = np.asarray([len(s.split()) for s in instructions], dtype=np.float32) if instructions else np.zeros((0,), dtype=np.float32)

    # repetition (clip/padding) analysis: mimic VLANavDataset._get_sequence local index clipping.
    # For episode mode, local_start is always 0.
    # For sliding_window mode, we sample local_start uniformly from the same range the dataset uses.
    rng = np.random.default_rng(seed)

    def sample_local_start(ep_len: int) -> int:
        if sample_mode == "episode":
            return 0
        max_start = ep_len - horizon + pad_before + pad_after
        # dataset uses: for i in range(max(1, max_start))
        n = max(1, max_start)
        return int(rng.integers(low=0, high=n))

    episode_indices = np.arange(len(lens))
    if max_episodes_for_dynamics is not None and len(episode_indices) > max_episodes_for_dynamics:
        episode_indices = rng.choice(episode_indices, size=max_episodes_for_dynamics, replace=False)

    repetition_fracs: List[float] = []
    repeated_tail_lens: List[int] = []
    step_deltas: List[float] = []
    action_minus_pos: List[float] = []

    for ep_idx in episode_indices:
        ep_start = int(episode_starts[ep_idx])
        ep_end = int(episode_ends[ep_idx])
        ep_len = int(lens[ep_idx])
        ls = sample_local_start(ep_len)

        local_indices = (np.arange(horizon, dtype=np.int64) + ls - pad_before)
        local_clipped = np.clip(local_indices, 0, ep_len - 1)
        global_idx = ep_start + local_clipped

        # repetition fraction in the action sequence due to clipping
        unique = int(np.unique(global_idx).shape[0])
        repetition_fracs.append(1.0 - (unique / float(horizon)))

        # repeated tail length (how many final timesteps are clipped to last index)
        last_local = ep_len - 1
        tail = int(np.sum(local_clipped == last_local))
        repeated_tail_lens.append(tail)

        # action dynamics on the sampled global indices
        seq_action = actions[global_idx]
        seq_pos = agent_pos[global_idx]

        d_action = seq_action[1:] - seq_action[:-1]
        step_deltas.append(float(np.linalg.norm(d_action, axis=-1).mean()))

        diff = seq_action - seq_pos
        action_minus_pos.append(float(np.linalg.norm(diff, axis=-1).mean()))

    rep = np.asarray(repetition_fracs, dtype=np.float32) if repetition_fracs else np.zeros((0,), dtype=np.float32)
    tail = np.asarray(repeated_tail_lens, dtype=np.float32) if repeated_tail_lens else np.zeros((0,), dtype=np.float32)
    step_d = np.asarray(step_deltas, dtype=np.float32) if step_deltas else np.zeros((0,), dtype=np.float32)
    a_minus_p = np.asarray(action_minus_pos, dtype=np.float32) if action_minus_pos else np.zeros((0,), dtype=np.float32)

    ps = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]

    return DatasetSummary(
        name=name,
        zarr_dir=zarr_dir,
        n_steps=int(states.shape[0]),
        n_episodes=int(lens.shape[0]),
        n_images=int(images.shape[0]),
        episode_len_min=int(lens.min()),
        episode_len_mean=float(lens.mean()),
        episode_len_max=int(lens.max()),
        episode_len_percentiles=_percentiles(lens, ps),
        img_dtype=str(images.dtype),
        img_min=img_min,
        img_max=img_max,
        action_min=float(act_sample.min()),
        action_max=float(act_sample.max()),
        action_mean=float(act_sample.mean()),
        action_std=float(act_sample.std()),
        instruction_count=int(len(instructions)),
        instruction_unique=int(len(set(instructions))),
        instruction_len_mean=float(instr_lens.mean()) if instr_lens.size else 0.0,
        instruction_len_p95=float(np.percentile(instr_lens, 95)) if instr_lens.size else 0.0,
        repetition_frac_mean=float(rep.mean()) if rep.size else 0.0,
        repetition_frac_p95=float(np.percentile(rep, 95)) if rep.size else 0.0,
        repeated_tail_len_mean=float(tail.mean()) if tail.size else 0.0,
        step_delta_l2_mean=float(step_d.mean()) if step_d.size else 0.0,
        step_delta_l2_p95=float(np.percentile(step_d, 95)) if step_d.size else 0.0,
        action_minus_pos_l2_mean=float(a_minus_p.mean()) if a_minus_p.size else 0.0,
        action_minus_pos_l2_p95=float(np.percentile(a_minus_p, 95)) if a_minus_p.size else 0.0,
    )


def print_summary(s: DatasetSummary):
    print("=" * 80)
    print(f"{s.name}")
    print(f"zarr: {s.zarr_dir}")
    print(f"steps: {s.n_steps}  episodes: {s.n_episodes}  images: {s.n_images}")
    print(f"episode_len min/mean/max: {s.episode_len_min} / {s.episode_len_mean:.3f} / {s.episode_len_max}")
    print("episode_len percentiles:")
    for p in sorted(s.episode_len_percentiles.keys()):
        print(f"  p{p:02d}: {s.episode_len_percentiles[p]:.0f}")

    print(f"img dtype: {s.img_dtype}  img min/max (sampled): {s.img_min:.1f} / {s.img_max:.1f}")
    print(
        "action min/max/mean/std (sampled): "
        f"{s.action_min:.6f} / {s.action_max:.6f} / {s.action_mean:.6f} / {s.action_std:.6f}"
    )

    print(
        "instruction count/unique: "
        f"{s.instruction_count} / {s.instruction_unique}  "
        f"len_mean={s.instruction_len_mean:.2f}w len_p95={s.instruction_len_p95:.2f}w"
    )

    print(
        "clip-induced repetition (for configured horizon/pads/sample_mode): "
        f"mean={s.repetition_frac_mean:.3f} p95={s.repetition_frac_p95:.3f}  "
        f"repeated_tail_len_mean={s.repeated_tail_len_mean:.1f}"
    )
    print(
        "action dynamics on sampled sequences: "
        f"step_delta_l2 mean={s.step_delta_l2_mean:.6f} p95={s.step_delta_l2_p95:.6f}  "
        f"|action-agent_pos| mean={s.action_minus_pos_l2_mean:.6f} p95={s.action_minus_pos_l2_p95:.6f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Path to dataset A split directory (train/) or dataset.zarr")
    ap.add_argument("--b", required=True, help="Path to dataset B split directory (train/) or dataset.zarr")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--pad-before", type=int, default=0)
    ap.add_argument("--pad-after", type=int, default=0)
    ap.add_argument("--sample-mode", choices=["episode", "sliding_window"], default="episode")
    ap.add_argument("--max-episodes", type=int, default=5000, help="Episodes to sample for repetition/dynamics stats")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    s1 = summarize_dataset(
        name=args.name_a,
        split_path=args.a,
        horizon=args.horizon,
        pad_before=args.pad_before,
        pad_after=args.pad_after,
        sample_mode=args.sample_mode,
        max_episodes_for_dynamics=args.max_episodes,
        seed=args.seed,
    )
    s2 = summarize_dataset(
        name=args.name_b,
        split_path=args.b,
        horizon=args.horizon,
        pad_before=args.pad_before,
        pad_after=args.pad_after,
        sample_mode=args.sample_mode,
        max_episodes_for_dynamics=args.max_episodes,
        seed=args.seed,
    )

    print_summary(s1)
    print_summary(s2)


if __name__ == "__main__":
    main()
