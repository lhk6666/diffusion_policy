"""
VLA Observation Encoder Adapter for Diffusion Policy

This module wraps the VLAEncoder to work with Diffusion Policy's interface.
It directly reuses VLAEncoder (now nn.Module) to ensure consistency with the VLA model.

Architecture:
  VLAEncoder (SigLIP + ConditionEncoder + CrossModalFusion)
       ↓
  context [B, N_v, d_model] → mean pool → [B, d_model]
       ↓
  Diffusion Policy UNet
"""

import sys
import os
from typing import Dict, Optional, List, Sequence, Union, Tuple
import torch
import torch.nn as nn

# Add FlowVLA to path for imports
flowvla_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
if flowvla_path not in sys.path:
    sys.path.insert(0, flowvla_path)

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

# Directly import VLAEncoder (now nn.Module, no Lightning dependency)
from src.models.encoder.encoder import VLAEncoder


class VLAObsEncoder(ModuleAttrMixin):
    """
    Adapter that wraps VLAEncoder for Diffusion Policy.
    
    Directly uses VLAEncoder (nn.Module) for exact consistency with FlowVLA.
    """
    
    def __init__(
        self,
        shape_meta: dict,
        # VLAEncoder parameters (same as VLA model)
        siglip_model_name: str = "google/siglip2-base-patch16-224",
        d_model: int = 768,
        num_heads: int = 8,
        num_fusion_layers: int = 4,
        unfreeze_last_n_layers: int = 0,
        # Default text for inference
        default_text: str = "navigate to the target",
        # Unused but for DP config compatibility
        imagenet_norm: bool = False,
    ):
        """
        Args:
            shape_meta: DP shape metadata
            siglip_model_name: Same as VLA model
            d_model: Same as VLA model (768)
            num_heads: Same as VLA model (8)
            num_fusion_layers: Same as VLA model (2)
            unfreeze_last_n_layers: Same as VLA model (0 = freeze all)
            default_text: Default text prompt for inference
        """
        super().__init__()
        
        self.shape_meta = shape_meta
        self.d_model = d_model
        self.default_text = default_text
        
        # Parse shape meta for DP compatibility
        self.rgb_keys = []
        self.low_dim_keys = []
        self.key_shape_map = {}
        
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            obs_type = attr.get('type', 'low_dim')
            self.key_shape_map[key] = shape
            
            if obs_type == 'rgb':
                self.rgb_keys.append(key)
            elif obs_type == 'low_dim':
                self.low_dim_keys.append(key)
        
        self.rgb_keys = sorted(self.rgb_keys)
        self.low_dim_keys = sorted(self.low_dim_keys)
        
        # ============================================
        # Directly use VLAEncoder (now nn.Module)
        # ============================================
        self.encoder = VLAEncoder(
            siglip_model_name=siglip_model_name,
            unfreeze_last_n_layers=unfreeze_last_n_layers,
            d_model=d_model,
            num_heads=num_heads,
            num_fusion_layers=num_fusion_layers
        )
        
        # Calculate output dim: d_model (from encoder) + low_dim features (direct concat)
        low_dim_size = sum(self.key_shape_map[k][-1] for k in self.low_dim_keys)
        self.low_dim_size = low_dim_size
        self.output_dim = d_model + low_dim_size
    
    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode observations for Diffusion Policy.
        
        Args:
            obs_dict: Dictionary with:
                - 'image': [B, C, H, W] RGB image, normalized to [-1, 1] (SigLIP format)
                - 'agent_pos': [B, 2] agent position (optional)
                - 'text': List[str] text prompts (optional)
        
        Returns:
            features: [B, output_dim] context embedding
        """
        batch_size = None
        features = []
        
        # Get image from obs_dict
        for key in self.rgb_keys:
            img = obs_dict[key]
            if batch_size is None:
                batch_size = img.shape[0]
            
            # Image should already be normalized to [-1, 1] by dataset
            # SigLIPBackbone.encode_image handles tensor input directly
            
            # Get text - use default if not provided
            texts = obs_dict.get('text', None)
            if texts is None:
                texts = [self.default_text] * batch_size
            # Forward through FlowVLA VLAEncoder.
            # Depending on FlowVLA version, `context` can be:
            # - token context: [B, N, D]
            # - pooled context: [B, D]
            context, _ = self.encoder(img, texts)
            if context.ndim == 3:
                context = context.mean(dim=1)
            elif context.ndim != 2:
                raise RuntimeError(f"Unexpected encoder context shape: {tuple(context.shape)}")
            features.append(context)
        
        # Process low_dim inputs (direct concat, no projection)
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            features.append(data)
        
        # Concatenate all features
        result = torch.cat(features, dim=-1)
        return result


class VLAObsEncoderV2(ModuleAttrMixin):
    """V2 adapter for Diffusion Policy that matches FlowVLA encoder contract.

    Key fixes vs `VLAObsEncoder`:
    - Uses `VLAEncoder`'s returned `context` directly (shape [B, d_model]).
    - Robustly expands text / token batches to match flattened image batches (B*To).
      Diffusion Policy flattens time for tensors, but leaves `text` as length-B list.
    """

    def __init__(
        self,
        shape_meta: dict,
        siglip_model_name: str = "google/siglip2-base-patch16-224",
        d_model: int = 768,
        num_heads: int = 8,
        num_fusion_layers: int = 2,
        unfreeze_last_n_layers: int = 0,
        siglip_deterministic_embeddings: bool = True,
        default_text: str = "navigate to the target",
        imagenet_norm: bool = False,
    ):
        super().__init__()

        self.shape_meta = shape_meta
        self.d_model = d_model
        self.default_text = default_text

        # Parse shape meta
        self.rgb_keys = []
        self.low_dim_keys = []
        self.key_shape_map = {}

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            obs_type = attr.get('type', 'low_dim')
            self.key_shape_map[key] = shape

            if obs_type == 'rgb':
                self.rgb_keys.append(key)
            elif obs_type == 'low_dim':
                self.low_dim_keys.append(key)

        self.rgb_keys = sorted(self.rgb_keys)
        self.low_dim_keys = sorted(self.low_dim_keys)

        self.encoder = VLAEncoder(
            siglip_model_name=siglip_model_name,
            unfreeze_last_n_layers=unfreeze_last_n_layers,
            siglip_deterministic_embeddings=siglip_deterministic_embeddings,
            d_model=d_model,
            num_heads=num_heads,
            num_fusion_layers=num_fusion_layers,
        )

        low_dim_size = sum(self.key_shape_map[k][-1] for k in self.low_dim_keys)
        self.low_dim_size = low_dim_size
        self.output_dim = d_model + low_dim_size

    @staticmethod
    def _expand_list_to_batch(texts: Sequence[str], target_batch: int) -> List[str]:
        if len(texts) == 0:
            return []
        if len(texts) == target_batch:
            return list(texts)
        if target_batch % len(texts) == 0:
            rep = target_batch // len(texts)
            return [t for t in texts for _ in range(rep)]
        if len(texts) == 1:
            return [texts[0]] * target_batch
        raise ValueError(
            f"Cannot expand texts batch of size {len(texts)} to match target_batch={target_batch}."
        )

    @staticmethod
    def _expand_tensor_to_batch(x: torch.Tensor, target_batch: int) -> torch.Tensor:
        if x.shape[0] == target_batch:
            return x
        if target_batch % x.shape[0] == 0:
            rep = target_batch // x.shape[0]
            return x.repeat_interleave(rep, dim=0)
        if x.shape[0] == 1:
            return x.expand(target_batch, *x.shape[1:])
        raise ValueError(
            f"Cannot expand tensor batch of size {x.shape[0]} to match target_batch={target_batch}."
        )

    def _get_condition_inputs(
        self,
        obs_dict: Dict[str, torch.Tensor],
        target_batch: int,
    ) -> Tuple[Union[torch.Tensor, str, List[str]], Optional[torch.Tensor]]:
        """Pick (texts_or_input_ids, attention_mask) and expand to `target_batch` if needed."""
        input_ids = obs_dict.get('input_ids', None)
        attention_mask = obs_dict.get('attention_mask', None)
        if input_ids is not None:
            if not torch.is_tensor(input_ids):
                raise TypeError("obs_dict['input_ids'] must be a torch.Tensor")
            input_ids = self._expand_tensor_to_batch(input_ids, target_batch)
            if attention_mask is not None:
                if not torch.is_tensor(attention_mask):
                    raise TypeError("obs_dict['attention_mask'] must be a torch.Tensor")
                attention_mask = self._expand_tensor_to_batch(attention_mask, target_batch)
            return input_ids, attention_mask

        texts = obs_dict.get('text', None)
        if texts is None:
            return [self.default_text] * target_batch, None
        if isinstance(texts, str):
            return [texts] * target_batch, None
        if isinstance(texts, (list, tuple)):
            return self._expand_list_to_batch(texts, target_batch), None

        raise TypeError("obs_dict['text'] must be str or list[str] when provided")

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = None
        features: List[torch.Tensor] = []

        for key in self.rgb_keys:
            img = obs_dict[key]
            if batch_size is None:
                batch_size = img.shape[0]

            texts_or_input_ids, attention_mask = self._get_condition_inputs(obs_dict, batch_size)
            context, _ = self.encoder(img, texts_or_input_ids, attention_mask=attention_mask)
            if context.ndim == 3:
                context = context.mean(dim=1)
            elif context.ndim != 2:
                raise RuntimeError(f"Unexpected encoder context shape: {tuple(context.shape)}")
            features.append(context)

        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            features.append(data)
        return torch.cat(features, dim=-1)

    @torch.no_grad()
    def output_shape(self):
        """Return output shape for DP compatibility."""
        return (self.output_dim,)
    
    def load_vla_encoder_weights(self, checkpoint_path: str):
        """
        Load encoder weights from a trained VLA model checkpoint.
        
        This allows initializing DP with pretrained VLA encoder weights.
        
        Args:
            checkpoint_path: Path to VLA model checkpoint (.ckpt)
        """
        print(f"Loading VLA encoder weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract encoder state dict
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Map VLAModel encoder weights to our encoder
        encoder_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                new_key = key[len('encoder.'):]
                encoder_dict[new_key] = value
        
        # Load into encoder
        if encoder_dict:
            missing, unexpected = self.encoder.load_state_dict(encoder_dict, strict=False)
            print(f"  Loaded {len(encoder_dict)} parameters")
            if missing:
                print(f"  Missing keys: {missing[:5]}..." if len(missing) > 5 else f"  Missing keys: {missing}")
            if unexpected:
                print(f"  Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"  Unexpected keys: {unexpected}")
        else:
            print("  Warning: No encoder weights found in checkpoint")


class VLAObsEncoderTokens(ModuleAttrMixin):
    """Adapter that returns *token* embeddings from VLAEncoder (no pooling).

    Output:
        tokens: (B, N_tokens, d_model)

    Notes:
    - This is intended for token-level conditioning (e.g., cross-attention memory),
      not for producing a single pooled feature vector.
    - Low-dim inputs are ignored by default to keep `cond_dim=d_model` clean.
    """

    def __init__(
        self,
        shape_meta: dict,
        siglip_model_name: str = "google/siglip2-base-patch16-224",
        d_model: int = 768,
        num_heads: int = 8,
        num_fusion_layers: int = 2,
        unfreeze_last_n_layers: int = 0,
        siglip_deterministic_embeddings: bool = True,
        default_text: str = "navigate to the target",
        imagenet_norm: bool = False,
    ):
        super().__init__()

        self.shape_meta = shape_meta
        self.d_model = d_model
        self.default_text = default_text

        # Parse shape meta
        self.rgb_keys: List[str] = []
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            obs_type = attr.get('type', 'low_dim')
            if obs_type == 'rgb':
                self.rgb_keys.append(key)
        self.rgb_keys = sorted(self.rgb_keys)
        if len(self.rgb_keys) == 0:
            raise ValueError("VLAObsEncoderTokens requires at least one rgb observation key")

        self.encoder = VLAEncoder(
            siglip_model_name=siglip_model_name,
            unfreeze_last_n_layers=unfreeze_last_n_layers,
            siglip_deterministic_embeddings=siglip_deterministic_embeddings,
            d_model=d_model,
            num_heads=num_heads,
            num_fusion_layers=num_fusion_layers,
        )

    @staticmethod
    def _expand_list_to_batch(texts: Sequence[str], target_batch: int) -> List[str]:
        if len(texts) == 0:
            return []
        if len(texts) == target_batch:
            return list(texts)
        if target_batch % len(texts) == 0:
            rep = target_batch // len(texts)
            return [t for t in texts for _ in range(rep)]
        if len(texts) == 1:
            return [texts[0]] * target_batch
        raise ValueError(
            f"Cannot expand texts batch of size {len(texts)} to match target_batch={target_batch}."
        )

    @staticmethod
    def _expand_tensor_to_batch(x: torch.Tensor, target_batch: int) -> torch.Tensor:
        if x.shape[0] == target_batch:
            return x
        if target_batch % x.shape[0] == 0:
            rep = target_batch // x.shape[0]
            return x.repeat_interleave(rep, dim=0)
        if x.shape[0] == 1:
            return x.expand(target_batch, *x.shape[1:])
        raise ValueError(
            f"Cannot expand tensor batch of size {x.shape[0]} to match target_batch={target_batch}."
        )

    def _get_condition_inputs(
        self,
        obs_dict: Dict[str, torch.Tensor],
        target_batch: int,
    ) -> Tuple[Union[torch.Tensor, str, List[str]], Optional[torch.Tensor]]:
        input_ids = obs_dict.get('input_ids', None)
        attention_mask = obs_dict.get('attention_mask', None)
        if input_ids is not None:
            if not torch.is_tensor(input_ids):
                raise TypeError("obs_dict['input_ids'] must be a torch.Tensor")
            input_ids = self._expand_tensor_to_batch(input_ids, target_batch)
            if attention_mask is not None:
                if not torch.is_tensor(attention_mask):
                    raise TypeError("obs_dict['attention_mask'] must be a torch.Tensor")
                attention_mask = self._expand_tensor_to_batch(attention_mask, target_batch)
            return input_ids, attention_mask

        texts = obs_dict.get('text', None)
        if texts is None:
            return [self.default_text] * target_batch, None
        if isinstance(texts, str):
            return [texts] * target_batch, None
        if isinstance(texts, (list, tuple)):
            return self._expand_list_to_batch(texts, target_batch), None
        raise TypeError("obs_dict['text'] must be str or list[str] when provided")

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = None
        token_batches: List[torch.Tensor] = []

        for key in self.rgb_keys:
            img = obs_dict[key]
            if batch_size is None:
                batch_size = img.shape[0]
            texts_or_input_ids, attention_mask = self._get_condition_inputs(obs_dict, batch_size)
            context, _ = self.encoder(img, texts_or_input_ids, attention_mask=attention_mask)
            # context: (B, N, d_model) preferred
            if context.ndim == 2:
                context = context.unsqueeze(1)
            elif context.ndim != 3:
                raise RuntimeError(f"Unexpected encoder context shape: {tuple(context.shape)}")
            token_batches.append(context)

        if len(token_batches) == 1:
            return token_batches[0]
        return torch.cat(token_batches, dim=1)


def test_encoder():
    """Quick test of the VLA encoder adapter."""
    import torch
    
    shape_meta = {
        'obs': {
            'image': {'shape': [3, 224, 224], 'type': 'rgb'},
            'agent_pos': {'shape': [2], 'type': 'low_dim'}
        },
        'action': {'shape': [2]}
    }
    
    print("Creating VLAObsEncoder...")
    encoder = VLAObsEncoder(
        shape_meta=shape_meta,
        siglip_model_name="google/siglip2-base-patch16-224",
        d_model=768,
        num_fusion_layers=2
    )
    encoder = encoder.cuda()
    
    print(f"Output dim: {encoder.output_dim}")
    print(f"Using VLAEncoder directly (nn.Module)")
    
    obs_dict = {
        'image': torch.randn(2, 3, 224, 224).cuda(),
        'agent_pos': torch.randn(2, 2).cuda(),
        'text': ["go to the chair", "navigate to the table"]
    }
    
    print("Forward pass...")
    output = encoder(obs_dict)
    print(f"Output shape: {output.shape}")
    print(f"Expected: [2, {encoder.output_dim}]")
    
    print("\n✅ VLAObsEncoder test passed!")


if __name__ == "__main__":
    test_encoder()
