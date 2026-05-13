from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator


class DiffusionTransformerVLAImageTokenPolicy(BaseImagePolicy):
    """Transformer diffusion policy conditioned on VLA *spatial tokens*.

    Conditioning:
        cond = encoder_tokens with shape (B, N_tokens, cond_dim)

    This avoids pooling, and avoids overloading n_obs_steps to mean token count.
    """

    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: nn.Module,
        model: TransformerForDiffusion,
        noise_scheduler: DDPMScheduler,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: Optional[int] = None,
        obs_as_cond: bool = True,
        pred_action_steps_only: bool = False,
        # Action-head supervision: scalar weight on the 4-way CE loss over
        # ``action_id``. See DiffusionUnetImagePolicy for the rationale on
        # the 0.1 default.
        action_loss_weight: float = 0.1,
        action_label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = int(action_shape[0])

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()

        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.action_loss_weight     = float(action_loss_weight)
        self.action_label_smoothing = float(action_label_smoothing)

        if not self.obs_as_cond:
            raise ValueError("DiffusionTransformerVLAImageTokenPolicy requires obs_as_cond=True")

    # ========= utilities =========
    @staticmethod
    def _split_tensor_and_nontensor(obs_dict: Dict):
        tensor_dict = {}
        other_dict = {}
        for k, v in obs_dict.items():
            if torch.is_tensor(v):
                tensor_dict[k] = v
            else:
                other_dict[k] = v
        return tensor_dict, other_dict

    def _normalize_obs(self, obs_dict: Dict) -> Dict:
        tensor_dict, other_dict = self._split_tensor_and_nontensor(obs_dict)
        if len(tensor_dict) > 0:
            tensor_dict = self.normalizer.normalize(tensor_dict)
        tensor_dict.update(other_dict)
        return tensor_dict

    @staticmethod
    def _flatten_time(obs_dict: Dict, To: int) -> Dict:
        out = {}
        for k, v in obs_dict.items():
            if torch.is_tensor(v):
                out[k] = v[:, :To, ...].reshape(-1, *v.shape[2:])
            else:
                out[k] = v
        return out

    # ========= inference =========
    def conditional_sample(self, condition_data, condition_mask, cond=None, generator=None, **kwargs):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(trajectory, t, cond)
            trajectory = scheduler.step(
                model_output,
                t,
                trajectory,
                generator=generator,
                **kwargs,
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self._normalize_obs(obs_dict)

        # infer batch/time from a tensor field
        tensor_val = next(v for v in nobs.values() if torch.is_tensor(v))
        B, _, *_ = tensor_val.shape
        To = self.n_obs_steps
        T = self.horizon
        Da = self.action_dim

        device = self.device
        dtype = self.dtype

        # encode tokens from first To observations (flatten time)
        encoder_in = self._flatten_time(nobs, To)
        tokens = self.obs_encoder(encoder_in)  # (B*To, N, D)
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected obs_encoder tokens (B*To,N,D), got {tuple(tokens.shape)}")

        BTo, N, D = tokens.shape
        if BTo != B * To:
            raise RuntimeError(f"Token batch mismatch: got {BTo}, expected {B*To}")

        # Treat as *token memory* (not temporal obs steps)
        cond = tokens.reshape(B, To * N, D)

        shape = (B, T, Da)
        if self.pred_action_steps_only:
            shape = (B, self.n_action_steps, Da)
        cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        nsample = self.conditional_sample(cond_data, cond_mask, cond=cond, **self.kwargs)

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        return {
            'action': action,
            'action_pred': action_pred,
        }

    # ========= training =========
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
        self,
        transformer_weight_decay: float,
        obs_encoder_weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(weight_decay=transformer_weight_decay)
        optim_groups.append({
            'params': self.obs_encoder.parameters(),
            'weight_decay': obs_encoder_weight_decay,
        })
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    def compute_loss(self, batch):
        assert 'valid_mask' not in batch

        # normalize obs tensors; keep non-tensor fields (e.g., text) for encoder
        nobs = self._normalize_obs(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        batch_size = nactions.shape[0]
        To = self.n_obs_steps

        # encode tokens from first To frames
        encoder_in = self._flatten_time(nobs, To)
        tokens = self.obs_encoder(encoder_in)  # (B*To, N, D)
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected obs_encoder tokens (B*To,N,D), got {tuple(tokens.shape)}")
        BTo, N, D = tokens.shape
        if BTo != batch_size * To:
            raise RuntimeError(f"Token batch mismatch: got {BTo}, expected {batch_size*To}")
        cond = tokens.reshape(batch_size, To * N, D)

        trajectory = nactions
        if self.pred_action_steps_only:
            start = To - 1
            end = start + self.n_action_steps
            trajectory = nactions[:, start:end]

        # generate inpainting mask (no conditioning on trajectory here)
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()

        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        pred = self.model(noisy_trajectory, timesteps, cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        # ── Action-head CE loss (mirrors SecVLA's ``status_head``) ────────
        # The encoder cached its 4-way logits during the forward above; we
        # just need the ``action_id`` target from the batch. ``-100``
        # entries are ignored by cross_entropy. Skip when the head is
        # disabled OR no action_id was provided OR the weight is ≤ 0.
        last_logits = getattr(self.obs_encoder, "last_action_logits", None)
        action_id   = batch.get("action_id", None)
        if (
            self.action_loss_weight > 0.0
            and last_logits is not None
            and action_id is not None
        ):
            # Encoder ran on B*To frames; with n_obs_steps=1 (the only
            # value the SecVLA DP dataset currently supplies action_id for)
            # the rows line up 1:1 with action_id. For To>1 we'd need
            # per-step ids; assert to surface that case loudly.
            if last_logits.shape[0] != action_id.shape[0]:
                raise RuntimeError(
                    f"action-head logits batch ({last_logits.shape[0]}) does not "
                    f"match action_id batch ({action_id.shape[0]}). Likely caused "
                    f"by n_obs_steps>1 without per-step action_id supervision."
                )
            target_aid = action_id.long().view(-1)
            valid = target_aid != -100
            if int(valid.sum().item()) > 0:
                action_loss = F.cross_entropy(
                    last_logits, target_aid,
                    ignore_index=-100,
                    label_smoothing=self.action_label_smoothing,
                )
                loss = loss + self.action_loss_weight * action_loss

        return loss
