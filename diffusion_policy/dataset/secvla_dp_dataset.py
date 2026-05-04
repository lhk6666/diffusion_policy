"""SecVLA zarr dataset wrapper for Diffusion Policy baselines.

Mirrors the current single-frame SecVLA pipeline (memory module disabled):

    pixel_values        [1, 3, H, W]
    input_ids           [1, L]
    attention_mask      [1, L]
    depth               [1, 1, H, W]          (optional)
    depth_valid_mask    [1, 1, H, W]          (optional)

and uses the patched baseline supervision:

    dp_traj_sector_delta/{sample_id}[flow_idx] -> [H, 2]

Each DP sample is one SecVLA `(sample_id, flow_idx)` item with a single
observation step (`n_obs_steps = 1`) and a full delta-XY trajectory target.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import zarr

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


def _find_workspace_root(start: Path) -> Path:
    for cand in [start, *start.parents]:
        if (cand / "SecVLA").is_dir():
            return cand
    raise RuntimeError(f"Could not locate workspace root from {start}")


_WORKSPACE_ROOT = _find_workspace_root(Path(__file__).resolve())
_SECVLA_ROOT = _WORKSPACE_ROOT / "SecVLA"
if str(_SECVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(_SECVLA_ROOT))

from src.dataset_loader import SecVLADataset  # noqa: E402


def _resolve_split_dir(path: str) -> Path:
    p = Path(path)
    if p.is_dir() and p.name == "dataset.zarr":
        return p.parent
    nested = p / "dataset.zarr"
    if p.is_dir() and nested.is_dir():
        return p
    if p.parent.is_dir() and p.name == "dataset.zarr":
        return p.parent
    return p


def _identity_normalizer(dtype: torch.dtype = torch.float32) -> SingleFieldLinearNormalizer:
    return SingleFieldLinearNormalizer.create_identity(dtype=dtype)


class SecVLADeltaTrajectoryDataset(BaseImageDataset):
    """DP training dataset that reuses SecVLA preprocessing exactly."""

    def __init__(
        self,
        zarr_dir: str,
        horizon: int = 100,
        instruction_mode: str = "sub_instruction",
        siglip_model_name: str = "google/siglip2-base-patch16-224",
        max_text_length: int = 64,
        use_depth: bool = True,
        require_depth: bool = True,
        depth_zmax: float = 5.0,
    ):
        super().__init__()

        if instruction_mode not in {"instruction", "sub_instruction"}:
            raise ValueError(
                f"instruction_mode must be 'instruction' or 'sub_instruction', got {instruction_mode!r}"
            )

        self.split_dir = _resolve_split_dir(zarr_dir)
        self.horizon = int(horizon)
        self.instruction_mode = str(instruction_mode)
        self.siglip_model_name = str(siglip_model_name)
        self.max_text_length = int(max_text_length)
        self.use_depth = bool(use_depth)
        self.require_depth = bool(require_depth)
        self.depth_zmax = float(depth_zmax)

        self.base = SecVLADataset(
            data_dir=str(self.split_dir),
            if_test=False,
            siglip_model_name=siglip_model_name,
            max_text_length=max_text_length,
            use_depth=use_depth,
            require_depth=require_depth,
        )

        zarr_path = self.split_dir / "dataset.zarr"
        self._store = zarr.open(str(zarr_path), mode="r")
        self._episodes_grp = self._store.get("episodes", None)
        if self._episodes_grp is None:
            raise KeyError(
                f"{zarr_path} is missing 'episodes/' group — "
                "this adapter requires v3.1 episode-major datasets."
            )
        first_uid = next(iter(self._episodes_grp.group_keys()), None)
        if first_uid is None or "dp_traj_sector_delta" not in self._episodes_grp[first_uid]:
            raise KeyError(
                f"episodes/{first_uid}/dp_traj_sector_delta not found in {zarr_path}. "
                "Run SecVLA/scripts/datasets/patch_baseline_labels.py first."
            )

        self._action_normalizer: Optional[LinearNormalizer] = None

    def __len__(self) -> int:
        return len(self.base)

    def _load_action_delta(self, idx: int) -> torch.Tensor:
        item = self.base.items[idx]
        uid = item["episode_uid"]
        t_local = int(item["t_local"])
        k = int(item["k"])
        delta = np.asarray(
            self._episodes_grp[uid]["dp_traj_sector_delta"][t_local, k],
            dtype=np.float32,
        )
        if delta.ndim != 2 or delta.shape[-1] != 2:
            raise ValueError(
                f"dp_traj_sector_delta[{uid}][{t_local},{k}] must be [H,2], got {tuple(delta.shape)}"
            )
        if int(delta.shape[0]) != self.horizon:
            raise ValueError(
                f"Expected horizon={self.horizon}, but dp_traj_sector_delta[{uid}][{t_local},{k}] "
                f"has shape {tuple(delta.shape)}"
            )
        return torch.from_numpy(delta)

    def _select_text(self, item: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.instruction_mode == "sub_instruction":
            has_sub = bool(item.get("has_sub_instruction", torch.tensor(False)).item())
            if has_sub:
                return item["sub_input_ids"], item["sub_attention_mask"]

        input_ids = item["input_ids"]
        attention_mask = item.get("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.base[idx]
        pixel_values = item["pixel_values"]  # [3, H, W]
        input_ids, attention_mask = self._select_text(item)
        action = self._load_action_delta(idx)

        obs: Dict[str, torch.Tensor] = {
            "pixel_values": pixel_values.unsqueeze(0),
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
        }

        if "depth" in item:
            obs["depth"] = item["depth"].unsqueeze(0)
            obs["depth_valid_mask"] = item["depth_valid_mask"].unsqueeze(0)

        return {
            "obs": obs,
            "action": action,
        }

    def _compute_action_stats(self) -> dict:
        mins = np.full((2,), np.inf, dtype=np.float64)
        maxs = np.full((2,), -np.inf, dtype=np.float64)
        sums = np.zeros((2,), dtype=np.float64)
        sumsq = np.zeros((2,), dtype=np.float64)
        count = 0

        # Group items by episode so each episode's [T,K,H,2] array is fetched
        # exactly once instead of one zarr read per (t_local, k) item.
        items_by_uid: Dict[str, list] = {}
        for item in self.base.items:
            items_by_uid.setdefault(item["episode_uid"], []).append(
                (int(item["t_local"]), int(item["k"]))
            )

        for uid, tk_pairs in items_by_uid.items():
            arr = np.asarray(
                self._episodes_grp[uid]["dp_traj_sector_delta"][:],
                dtype=np.float64,
            )  # [T, K, H, 2]
            idx_t = np.fromiter((t for t, _ in tk_pairs), dtype=np.int64, count=len(tk_pairs))
            idx_k = np.fromiter((k for _, k in tk_pairs), dtype=np.int64, count=len(tk_pairs))
            gathered = arr[idx_t, idx_k]  # [N_items, H, 2] — same slices as the per-item path
            flat = gathered.reshape(-1, 2)
            mins = np.minimum(mins, flat.min(axis=0))
            maxs = np.maximum(maxs, flat.max(axis=0))
            sums += flat.sum(axis=0)
            sumsq += np.square(flat).sum(axis=0)
            count += int(flat.shape[0])

        if count <= 0:
            raise RuntimeError("No action labels found while computing DP normalizer.")

        mean = sums / float(count)
        var = np.maximum(sumsq / float(count) - np.square(mean), 0.0)
        std = np.sqrt(var)
        return {
            "min": mins.astype(np.float32),
            "max": maxs.astype(np.float32),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
        }

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        if mode != "limits":
            raise ValueError("SecVLADeltaTrajectoryDataset currently only supports mode='limits'")
        if self._action_normalizer is not None:
            return self._action_normalizer

        stats = self._compute_action_stats()
        input_min = stats["min"]
        input_max = stats["max"]
        input_range = input_max - input_min
        ignore_dim = input_range < 1e-4
        input_range[ignore_dim] = 2.0
        scale = (2.0 / input_range).astype(np.float32)
        offset = (-1.0 - scale * input_min).astype(np.float32)
        offset[ignore_dim] = 0.0

        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_manual(
            scale=scale,
            offset=offset,
            input_stats_dict=stats,
        )

        # Identity normalizers keep the exact SecVLA encoder contract intact.
        for key in ("pixel_values", "input_ids", "attention_mask", "depth", "depth_valid_mask"):
            normalizer[key] = _identity_normalizer()

        self._action_normalizer = normalizer
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.stack([self._load_action_delta(i) for i in range(len(self))], dim=0)

    def get_validation_dataset(self) -> "SecVLADeltaTrajectoryDataset":
        # Single-split dataset by default. The *FromPath wrapper below loads a dedicated val split.
        return SecVLADeltaTrajectoryDataset(
            zarr_dir=str(self.split_dir),
            horizon=self.horizon,
            instruction_mode=self.instruction_mode,
            siglip_model_name=self.siglip_model_name,
            max_text_length=self.max_text_length,
            use_depth=self.use_depth,
            require_depth=self.require_depth,
            depth_zmax=self.depth_zmax,
        )


class SecVLADeltaTrajectoryDatasetFromPath(SecVLADeltaTrajectoryDataset):
    """Train/val split wrapper matching existing DP config patterns."""

    def __init__(
        self,
        train_path: str,
        val_path: Optional[str] = None,
        **kwargs,
    ):
        self._train_path = str(_resolve_split_dir(train_path))
        self._val_path = str(_resolve_split_dir(val_path)) if val_path is not None else None
        self._val_dataset: Optional[SecVLADeltaTrajectoryDataset] = None
        super().__init__(zarr_dir=self._train_path, **kwargs)

    def get_validation_dataset(self) -> SecVLADeltaTrajectoryDataset:
        if self._val_dataset is not None:
            return self._val_dataset
        if self._val_path is None:
            return super().get_validation_dataset()

        self._val_dataset = SecVLADeltaTrajectoryDataset(
            zarr_dir=self._val_path,
            horizon=self.horizon,
            instruction_mode=self.instruction_mode,
            siglip_model_name=self.siglip_model_name,
            max_text_length=self.max_text_length,
            use_depth=self.use_depth,
            require_depth=self.require_depth,
            depth_zmax=self.depth_zmax,
        )
        return self._val_dataset
