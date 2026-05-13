"""Exact SecVLA encoder adapters for Diffusion Policy.

Mirrors the current single-frame SecVLA pipeline (memory module disabled):

    pixel_values + text token ids + optional depth

This lets the DP baseline change only the decoder while reusing the same
encoder implementation as SecVLA.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


def _find_workspace_root(start: Path) -> Path:
    for cand in [start, *start.parents]:
        if (cand / "SecVLA").is_dir():
            return cand
    raise RuntimeError(f"Could not locate workspace root from {start}")


_WORKSPACE_ROOT = _find_workspace_root(Path(__file__).resolve())
_SECVLA_ROOT = _WORKSPACE_ROOT / "SecVLA"
if str(_SECVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(_SECVLA_ROOT))

from src.models.encoder.encoder import VLAEncoder  # noqa: E402
from src.models.encoder.action_head import ActionQueryDecoder  # noqa: E402


def _maybe_bool(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.to(dtype=torch.bool)


def _maybe_long(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.to(dtype=torch.long)


class _BaseSecVLAObsEncoder(ModuleAttrMixin):
    def __init__(
        self,
        siglip_model_name: str = "google/siglip2-base-patch16-224",
        d_model: int = 768,
        num_heads: int = 8,
        num_fusion_layers: int = 4,
        unfreeze_last_n_layers: int = 0,
        siglip_deterministic_embeddings: bool = True,
        use_depth: bool = True,
        depth_use_mask_channel: bool = True,
        sector_zmin: float = 0.0,
        sector_zmax: float = 5.0,
        expected_memory_size: Optional[int] = 1,
        encoder_checkpoint_path: Optional[str] = None,
        # ── Memory module knobs (match SecVLA) ──────────────────────────
        # ``use_memory=True`` engages the memory grounding path even with
        # single-frame inputs (the encoder wraps [B,3,H,W] as 1-slot
        # memory). Lets DP training stay architecturally consistent with
        # SecVLA's full model even when the dataset doesn't surface past
        # frames yet.
        use_memory: bool = False,
        n_vis_pool: int = 32,
        perceiver_depth: int = 1,
        # ── Action / status head (4-way: STOP / FWD / LEFT / RIGHT) ─────
        # Same module SecVLA uses as ``status_head``. The encoder caches
        # the latest logits on ``self.last_action_logits`` after every
        # forward so the policy's ``compute_loss`` can pick them up
        # without re-running the encoder.
        attach_action_head:   bool = False,
        num_actions:          int   = 4,
        action_head_layers:   int   = 2,
        action_head_dropout:  float = 0.1,
    ):
        super().__init__()

        self.d_model = int(d_model)
        self.encoder = VLAEncoder(
            siglip_model_name=siglip_model_name,
            unfreeze_last_n_layers=unfreeze_last_n_layers,
            siglip_deterministic_embeddings=siglip_deterministic_embeddings,
            use_depth=use_depth,
            use_memory=bool(use_memory),
            depth_use_mask_channel=depth_use_mask_channel,
            d_model=d_model,
            num_heads=num_heads,
            num_fusion_layers=num_fusion_layers,
            sector_zmax=sector_zmax,
            sector_zmin=sector_zmin,
            expected_memory_size=expected_memory_size,
            n_vis_pool=int(n_vis_pool),
            perceiver_depth=int(perceiver_depth),
        )

        if bool(attach_action_head):
            self.action_head = ActionQueryDecoder(
                d_model     = d_model,
                num_actions = int(num_actions),
                num_heads   = num_heads,
                num_layers  = int(action_head_layers),
                dropout     = float(action_head_dropout),
            )
        else:
            self.action_head = None

        # Cache for the policy: latest action-head logits from a forward
        # pass. Plain attribute (not buffer) so it's not persisted.
        self._last_action_logits: Optional[torch.Tensor] = None

        if encoder_checkpoint_path:
            self.load_secvla_encoder_weights(encoder_checkpoint_path)

    # ------------------------------------------------------------------
    # Encoder forward that also runs the action head once. Returning
    # ``context`` lets each wrapper (pooled / token) format its features
    # without a second encoder pass.
    # ------------------------------------------------------------------
    def _encode_with_action_head(
        self,
        obs_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        enc_inputs = self._prepare_encoder_inputs(obs_dict)
        context, *_ = self.encoder(
            enc_inputs["pixel_values"],
            enc_inputs["input_ids"],
            attention_mask=enc_inputs["attention_mask"],
            depth=enc_inputs["depth"],
            depth_valid_mask=enc_inputs["depth_valid_mask"],
        )
        if self.action_head is not None:
            ctx_for_head = context.unsqueeze(1) if context.ndim == 2 else context
            self._last_action_logits = self.action_head(ctx_for_head)
        else:
            self._last_action_logits = None
        return context

    def _prepare_encoder_inputs(self, obs_dict: Dict[str, torch.Tensor]) -> dict:
        pixel_values = obs_dict["pixel_values"]
        if pixel_values.dim() == 5:
            # Defensive: collapse [B, To, 3, H, W] from n_obs_steps>1 callers.
            bsz, to = int(pixel_values.shape[0]), int(pixel_values.shape[1])
            pixel_values = pixel_values.reshape(bsz * to, *pixel_values.shape[2:])
        if pixel_values.dim() != 4:
            raise ValueError(
                f"pixel_values must be [B,3,H,W], got {tuple(pixel_values.shape)}"
            )

        input_ids = _maybe_long(obs_dict.get("input_ids"))
        attention_mask = _maybe_long(obs_dict.get("attention_mask"))
        depth = obs_dict.get("depth", None)
        depth_valid_mask = _maybe_bool(obs_dict.get("depth_valid_mask", None))

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "depth": depth,
            "depth_valid_mask": depth_valid_mask,
        }

    def load_secvla_encoder_weights(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        encoder_dict = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                encoder_dict[key[len("encoder."):]] = value

        if not encoder_dict:
            raise KeyError(f"No `encoder.*` weights found in {checkpoint_path}")

        missing, unexpected = self.encoder.load_state_dict(encoder_dict, strict=False)

        hparams = checkpoint.get("hyper_parameters", {})
        expected_memory_size = hparams.get("expected_memory_size", None)
        if expected_memory_size is not None and hasattr(self.encoder, "set_expected_memory_size"):
            self.encoder.set_expected_memory_size(int(expected_memory_size))

        print(f"[SecVLAObsEncoder] loaded encoder weights from {checkpoint_path}")
        if missing:
            print(f"[SecVLAObsEncoder] missing keys ({len(missing)}): {missing[:8]}")
        if unexpected:
            print(f"[SecVLAObsEncoder] unexpected keys ({len(unexpected)}): {unexpected[:8]}")


class SecVLAObsEncoder(ModuleAttrMixin):
    """Pooled SecVLA encoder for UNet-style global conditioning."""

    def __init__(self, shape_meta: dict, pooling: str = "mean", **kwargs):
        super().__init__()
        self.shape_meta = shape_meta
        self.pooling = str(pooling)
        self.impl = _BaseSecVLAObsEncoder(**kwargs)
        self.output_dim = self.impl.d_model

    @property
    def encoder(self) -> VLAEncoder:
        return self.impl.encoder

    @property
    def action_head(self):
        return self.impl.action_head

    @property
    def last_action_logits(self) -> Optional[torch.Tensor]:
        """Logits from the action head on the most recent forward pass.
        ``None`` when the head is disabled. Read by the policy's
        ``compute_loss`` to add the CE term without re-running the encoder.
        """
        return self.impl._last_action_logits

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        context = self.impl._encode_with_action_head(obs_dict)
        if context.ndim == 2:
            return context
        if context.ndim != 3:
            raise RuntimeError(f"Unexpected SecVLA context shape: {tuple(context.shape)}")
        if self.pooling == "mean":
            return context.mean(dim=1)
        if self.pooling == "first":
            return context[:, 0]
        raise ValueError(f"Unsupported pooling={self.pooling!r}")

    @torch.no_grad()
    def output_shape(self):
        return (self.output_dim,)

    def load_secvla_encoder_weights(self, checkpoint_path: str) -> None:
        self.impl.load_secvla_encoder_weights(checkpoint_path)


class SecVLAObsEncoderTokens(ModuleAttrMixin):
    """Token-level SecVLA encoder for transformer cross-attention conditioning."""

    def __init__(self, shape_meta: dict, **kwargs):
        super().__init__()
        self.shape_meta = shape_meta
        self.impl = _BaseSecVLAObsEncoder(**kwargs)
        self.output_dim = self.impl.d_model

    @property
    def encoder(self) -> VLAEncoder:
        return self.impl.encoder

    @property
    def action_head(self):
        return self.impl.action_head

    @property
    def last_action_logits(self) -> Optional[torch.Tensor]:
        return self.impl._last_action_logits

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        context = self.impl._encode_with_action_head(obs_dict)
        if context.ndim == 2:
            context = context.unsqueeze(1)
        if context.ndim != 3:
            raise RuntimeError(f"Unexpected SecVLA context shape: {tuple(context.shape)}")
        return context

    def load_secvla_encoder_weights(self, checkpoint_path: str) -> None:
        self.impl.load_secvla_encoder_weights(checkpoint_path)
