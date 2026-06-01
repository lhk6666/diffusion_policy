"""SecVLA zarr dataset wrapper for Diffusion Policy baselines.

Mirrors the current single-frame SecVLA pipeline (memory module disabled):

    pixel_values        [1, 3, H, W]
    input_ids           [1, L]
    attention_mask      [1, L]
    depth               [1, 1, H, W]          (optional)
    depth_valid_mask    [1, 1, H, W]          (optional)

and uses the builder's BEV-augmented supervision:

    dp_traj_cart_delta/{sample_id}[t, k] -> [H, 2]   cart (x_fwd, y_lft) metres

Each DP sample is one SecVLA `(sample_id, flow_idx)` item with a single
observation step (`n_obs_steps = 1`) and a full delta-XY trajectory target.
The cart-metre deltas are divided by ``v_norm`` (read from the dataset's
``metadata.json:bev_grid.v_norm``, defaults to ``bev_x_max``) at load time
so DP's training target lives in the same canonical numeric range as
SecVLA's velocity-field supervision (``v_canonical = v_metric / v_norm``).
``dp_inference.py`` multiplies back by ``v_norm`` to recover cart metres.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

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


def _coerce_path_list(paths: Union[str, Sequence[str], None]) -> List[str]:
    """Accept ``str``, ``list[str]``, ``tuple[str]``, or ``None`` from yaml.
    Returns a list of resolved split dirs (possibly empty for ``None``).
    """
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        return [str(_resolve_split_dir(str(paths)))]
    out: List[str] = []
    for p in paths:
        if p is None:
            continue
        out.append(str(_resolve_split_dir(str(p))))
    return out


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
        # Waypoint baseline: collapse the full H-step delta trajectory into a
        # single endpoint (cumulative sum = goal displacement). Returns a
        # length-1 action so the DP transformer trains with horizon=1.
        waypoint_target: bool = False,
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
        self.waypoint_target = bool(waypoint_target)

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
        if first_uid is None or "dp_traj_cart_delta" not in self._episodes_grp[first_uid]:
            raise KeyError(
                f"episodes/{first_uid}/dp_traj_cart_delta not found in {zarr_path}. "
                "Expected the BEV-augmented dataset produced by "
                "SecVLA/scripts/datasets/data_augmentation."
            )

        # ``v_norm`` from metadata.json — matches the value SecVLA's decoder
        # buffers and ``V_bev`` carry. Fall back to ``bev_x_max``, then to
        # 5.0 (the historical default) when neither is recorded so legacy
        # exports keep working.
        self.v_norm: float = self._read_v_norm(self.split_dir)

        self._action_normalizer: Optional[LinearNormalizer] = None

    @staticmethod
    def _read_v_norm(split_dir: Path) -> float:
        meta_path = split_dir / "metadata.json"
        if not meta_path.is_file():
            return 5.0
        with open(meta_path, "r") as f:
            meta = json.load(f)
        bev = meta.get("bev_grid", {}) or {}
        for key in ("v_norm", "bev_x_max"):
            val = bev.get(key)
            if val is not None:
                return float(val)
        return 5.0

    def __len__(self) -> int:
        return len(self.base)

    def _load_action_delta(self, idx: int) -> torch.Tensor:
        item = self.base.items[idx]
        uid = item["episode_uid"]
        t_local = int(item["t_local"])
        k = int(item["k"])
        delta = np.asarray(
            self._episodes_grp[uid]["dp_traj_cart_delta"][t_local, k],
            dtype=np.float32,
        )
        if delta.ndim != 2 or delta.shape[-1] != 2:
            raise ValueError(
                f"dp_traj_cart_delta[{uid}][{t_local},{k}] must be [H,2], got {tuple(delta.shape)}"
            )
        if self.waypoint_target:
            # Endpoint = cumulative sum of all per-step deltas (goal
            # displacement from the current pose). Shape [1, 2] → horizon=1.
            delta = delta.sum(axis=0, keepdims=True)
        elif int(delta.shape[0]) != self.horizon:
            raise ValueError(
                f"Expected horizon={self.horizon}, but dp_traj_cart_delta[{uid}][{t_local},{k}] "
                f"has shape {tuple(delta.shape)}"
            )
        # Match SecVLA's velocity-field convention: predict canonical
        # ``cart / v_norm`` (so per-step deltas live around ±0.02 instead
        # of ±0.1m). Inference multiplies back by v_norm.
        delta = delta / float(self.v_norm)
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

        # Top-level action_id (kept outside ``obs`` so the LinearNormalizer
        # leaves it alone). Falls back to ACTION_IGNORE_INDEX (-100) for
        # legacy items missing the field; cross-entropy ignores those.
        action_id = item.get("action_id")
        if action_id is None:
            action_id = torch.tensor(-100, dtype=torch.long)
        else:
            action_id = action_id.long()

        return {
            "obs": obs,
            "action": action,
            "action_id": action_id,
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

        inv_v_norm = 1.0 / float(self.v_norm)
        for uid, tk_pairs in items_by_uid.items():
            arr = np.asarray(
                self._episodes_grp[uid]["dp_traj_cart_delta"][:],
                dtype=np.float64,
            ) * inv_v_norm  # [T, K, H, 2] — canonical units, matches _load_action_delta
            idx_t = np.fromiter((t for t, _ in tk_pairs), dtype=np.int64, count=len(tk_pairs))
            idx_k = np.fromiter((k for _, k in tk_pairs), dtype=np.int64, count=len(tk_pairs))
            gathered = arr[idx_t, idx_k]  # [N_items, H, 2] — same slices as the per-item path
            if self.waypoint_target:
                flat = gathered.sum(axis=1)  # [N_items, 2] cumulative endpoints
            else:
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
        # Persist v_norm with the normalizer state so it survives the
        # checkpoint and ``dp_inference.py`` can multiply back. ``create_manual``
        # asserts every input_stats value has the same shape as ``scale`` so
        # we broadcast the scalar to (2,) instead of registering a scalar.
        return {
            "min":    mins.astype(np.float32),
            "max":    maxs.astype(np.float32),
            "mean":   mean.astype(np.float32),
            "std":    std.astype(np.float32),
            "v_norm": np.full((2,), float(self.v_norm), dtype=np.float32),
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
            waypoint_target=self.waypoint_target,
        )


class SecVLADeltaTrajectoryDatasetFromPath(BaseImageDataset):
    """Multi-path train/val wrapper matching SecVLA's mixed-corpus training.

    Accepts a single path or a list of paths for both ``train_path`` and
    ``val_path``. Internally holds one ``SecVLADeltaTrajectoryDataset`` per
    path and routes ``__getitem__`` through a flat ``(child_idx,
    item_idx)`` index, so a ``DataLoader`` sees one contiguous dataset.

    All child datasets must agree on ``v_norm`` (otherwise their canonical
    deltas live in different numeric ranges and the shared normalizer is
    meaningless). The wrapper enforces this at construction.

    The normalizer's stats (min / max / mean / std / v_norm) are computed
    across ALL training paths together — same convention SecVLA's
    ``create_dataloader`` follows when mixing R2R + RxR.
    """

    def __init__(
        self,
        train_path: Union[str, Sequence[str]],
        val_path: Union[str, Sequence[str], None] = None,
        **kwargs,
    ):
        super().__init__()

        self._train_paths: List[str] = _coerce_path_list(train_path)
        if not self._train_paths:
            raise ValueError("train_path must be a non-empty str or list of strs")
        self._val_paths: List[str] = _coerce_path_list(val_path)
        self._child_kwargs: Dict = dict(kwargs)

        self._train_datasets: List[SecVLADeltaTrajectoryDataset] = [
            SecVLADeltaTrajectoryDataset(zarr_dir=p, **kwargs)
            for p in self._train_paths
        ]

        # Enforce v_norm consistency — canonical units only line up across
        # paths when v_norm is identical.
        v_norms = {ds.v_norm for ds in self._train_datasets}
        if len(v_norms) > 1:
            raise ValueError(
                f"Train paths disagree on v_norm: {sorted(v_norms)}. "
                "Re-export the offending split or split them into separate runs."
            )
        self.v_norm: float = float(next(iter(v_norms)))

        # Flat index: list of (child_idx, item_idx) so DataLoader sees one
        # contiguous dataset across all paths.
        self._index: List[tuple] = []
        for di, ds in enumerate(self._train_datasets):
            n = len(ds)
            self._index.extend((di, ii) for ii in range(n))

        # Borrow horizon from the first child so the workspace's eval path
        # (which reads ``dataset.horizon``) keeps working.
        first = self._train_datasets[0]
        self.horizon: int = int(first.horizon)
        self.instruction_mode: str = first.instruction_mode
        self.siglip_model_name: str = first.siglip_model_name
        self.max_text_length: int = first.max_text_length
        self.use_depth: bool = first.use_depth
        self.require_depth: bool = first.require_depth
        self.depth_zmax: float = first.depth_zmax
        self.waypoint_target: bool = first.waypoint_target

        self._action_normalizer: Optional[LinearNormalizer] = None
        self._val_dataset: Optional["SecVLADeltaTrajectoryDatasetFromPath"] = None

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        di, ii = self._index[int(idx)]
        return self._train_datasets[di][ii]

    # ------------------------------------------------------------------
    # Normalizer — aggregated across all train paths
    # ------------------------------------------------------------------

    def _compute_action_stats_multi(self) -> dict:
        mins = np.full((2,), np.inf, dtype=np.float64)
        maxs = np.full((2,), -np.inf, dtype=np.float64)
        sums = np.zeros((2,), dtype=np.float64)
        sumsq = np.zeros((2,), dtype=np.float64)
        count = 0
        inv_v_norm = 1.0 / float(self.v_norm)

        for child in self._train_datasets:
            items_by_uid: Dict[str, list] = {}
            for it in child.base.items:
                items_by_uid.setdefault(it["episode_uid"], []).append(
                    (int(it["t_local"]), int(it["k"]))
                )

            for uid, tk_pairs in items_by_uid.items():
                arr = np.asarray(
                    child._episodes_grp[uid]["dp_traj_cart_delta"][:],
                    dtype=np.float64,
                ) * inv_v_norm  # [T, K, H, 2] in canonical units
                idx_t = np.fromiter((t for t, _ in tk_pairs), dtype=np.int64, count=len(tk_pairs))
                idx_k = np.fromiter((k for _, k in tk_pairs), dtype=np.int64, count=len(tk_pairs))
                gathered = arr[idx_t, idx_k]
                if self.waypoint_target:
                    flat = gathered.sum(axis=1)  # [N_items, 2] cumulative endpoints
                else:
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
            "min":    mins.astype(np.float32),
            "max":    maxs.astype(np.float32),
            "mean":   mean.astype(np.float32),
            "std":    std.astype(np.float32),
            "v_norm": np.full((2,), float(self.v_norm), dtype=np.float32),
        }

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        if mode != "limits":
            raise ValueError("SecVLADeltaTrajectoryDatasetFromPath only supports mode='limits'")
        if self._action_normalizer is not None:
            return self._action_normalizer

        stats = self._compute_action_stats_multi()
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
        for key in ("pixel_values", "input_ids", "attention_mask", "depth", "depth_valid_mask"):
            normalizer[key] = _identity_normalizer()

        self._action_normalizer = normalizer
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.stack(
            [self[i]["action"] for i in range(len(self))],
            dim=0,
        )

    # ------------------------------------------------------------------
    # Validation: same multi-path wrapper rebuilt from the val splits.
    # Falls back to self when no val_path was provided (legacy behaviour).
    # ------------------------------------------------------------------

    def get_validation_dataset(self) -> "SecVLADeltaTrajectoryDatasetFromPath":
        if self._val_dataset is not None:
            return self._val_dataset
        if not self._val_paths:
            return self  # type: ignore[return-value]
        # Reuse the same class with val paths as the "train" list, no val.
        self._val_dataset = SecVLADeltaTrajectoryDatasetFromPath(
            train_path=self._val_paths,
            val_path=None,
            **self._child_kwargs,
        )
        return self._val_dataset
