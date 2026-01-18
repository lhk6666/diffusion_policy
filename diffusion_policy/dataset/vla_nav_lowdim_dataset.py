"""VLA Navigation low-dim dataset for Diffusion Policy.

This is a low-dimensional counterpart of `vla_nav_dataset.py`.

- Loads trajectories from a self-contained Zarr store.
- Outputs low-dim `obs` (agent_pos) and `action` sequences.

Expected Zarr structure:
- data/
  - action: (N_steps, 2) float32
  - agent_pos: (N_steps, 2) float32
- meta/
  - episode_ends: (E,) int64

This dataset intentionally ignores images/text, enabling use with
`DiffusionTransformerLowdimPolicy` and `TrainDiffusionTransformerLowdimWorkspace`.
"""

from __future__ import annotations

from typing import Dict, Optional, List, Tuple

import copy
import pathlib

import numpy as np
import torch
import zarr

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


class VLANavLowdimDataset(BaseLowdimDataset):
    def __init__(
        self,
        zarr_path: str,
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
        sample_mode: str = "sliding_window",  # 'sliding_window' or 'episode'
    ):
        super().__init__()

        self.zarr_path = pathlib.Path(zarr_path)
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.sequence_length = horizon
        self.sample_mode = sample_mode

        # Open Zarr store
        self.root = zarr.open(str(self.zarr_path), mode="r")

        # Load arrays into memory for fast access
        print(f"Loading lowdim dataset from {zarr_path}...")
        self.actions = self.root["data"]["action"][:]
        self.agent_pos = self.root["data"]["agent_pos"][:]
        self.episode_ends = self.root["meta"]["episode_ends"][:]

        self.n_steps = len(self.agent_pos)
        self.n_episodes = len(self.episode_ends)

        # Compute episode starts
        self.episode_starts = np.concatenate([[0], self.episode_ends[:-1]])

        # Setup train/val split by episode
        rng = np.random.default_rng(seed)
        n_val = int(self.n_episodes * val_ratio)

        indices = np.arange(self.n_episodes)
        rng.shuffle(indices)

        val_indices = set(indices[:n_val])
        self.train_mask = np.array([i not in val_indices for i in range(self.n_episodes)])

        # Optionally limit training episodes
        if max_train_episodes is not None:
            train_indices = np.where(self.train_mask)[0]
            if len(train_indices) > max_train_episodes:
                rng.shuffle(train_indices)
                keep_indices = set(train_indices[:max_train_episodes])
                self.train_mask = np.array([i in keep_indices for i in range(self.n_episodes)])

        # Build sample indices for this split
        self._build_indices(self.train_mask)

    def _build_indices(self, episode_mask: np.ndarray) -> None:
        """Build list of valid (episode_idx, start_idx) pairs for sampling."""
        self.indices: List[Tuple[int, int]] = []

        for ep_idx in range(self.n_episodes):
            if not episode_mask[ep_idx]:
                continue

            if self.sample_mode == "episode":
                # One sample per episode, always start from 0
                self.indices.append((ep_idx, 0))
            else:
                ep_start = self.episode_starts[ep_idx]
                ep_end = self.episode_ends[ep_idx]
                ep_len = ep_end - ep_start

                max_start = ep_len - self.sequence_length + self.pad_before + self.pad_after
                for i in range(max(1, max_start)):
                    self.indices.append((ep_idx, i))

    def get_validation_dataset(self):
        """Get validation dataset with inverted mask (same zarr file)."""
        val_set = copy.copy(self)
        val_set.train_mask = ~self.train_mask
        val_set._build_indices(val_set.train_mask)
        return val_set

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        data = {
            "obs": self.agent_pos.astype(np.float32),
            "action": self.actions.astype(np.float32),
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.actions)

    def __len__(self) -> int:
        return len(self.indices)

    def _get_sequence(self, ep_idx: int, local_start: int) -> Dict[str, np.ndarray]:
        ep_start = self.episode_starts[ep_idx]
        ep_end = self.episode_ends[ep_idx]
        ep_len = ep_end - ep_start

        local_indices = np.arange(self.sequence_length) + local_start - self.pad_before
        local_indices_clipped = np.clip(local_indices, 0, ep_len - 1)
        global_indices = ep_start + local_indices_clipped

        seq_agent_pos = self.agent_pos[global_indices]
        seq_action = self.actions[global_indices]

        return {
            "obs": seq_agent_pos.astype(np.float32),
            "action": seq_action.astype(np.float32),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, local_start = self.indices[idx]
        data = self._get_sequence(ep_idx, local_start)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


class VLANavLowdimDatasetFromPath(VLANavLowdimDataset):
    """Low-dim VLA Navigation dataset that loads train/val from separate paths."""

    @staticmethod
    def _resolve_zarr_path(path: Optional[str]) -> Optional[str]:
        if path is None:
            return None
        p = pathlib.Path(path)
        if p.is_dir() and p.suffix != ".zarr":
            nested = p / "dataset.zarr"
            if nested.exists() and nested.is_dir():
                return str(nested)
        return str(p)

    def __init__(
        self,
        train_path: str,
        val_path: Optional[str] = None,
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        seed: int = 42,
        max_train_episodes: Optional[int] = None,
        sample_mode: str = "sliding_window",
    ):
        train_path = self._resolve_zarr_path(train_path)
        val_path = self._resolve_zarr_path(val_path)
        super().__init__(
            zarr_path=train_path,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            seed=seed,
            val_ratio=0.0,
            max_train_episodes=max_train_episodes,
            sample_mode=sample_mode,
        )
        self.val_path = val_path
        self._val_dataset = None

    def get_validation_dataset(self):
        if self._val_dataset is not None:
            return self._val_dataset
        if self.val_path is None:
            return super().get_validation_dataset()

        self._val_dataset = VLANavLowdimDataset(
            zarr_path=self.val_path,
            horizon=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            sample_mode=self.sample_mode,
        )
        return self._val_dataset
