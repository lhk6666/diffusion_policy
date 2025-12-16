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
from typing import Dict, Optional, List
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
        num_fusion_layers: int = 2,
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
            
            # Forward through VLAEncoder
            context, _ = self.encoder(img, texts)  # [B, N_v, d_model]
            
            # Pool over visual tokens to get [B, d_model]
            context_pooled = context.mean(dim=1)
            features.append(context_pooled)
        
        # Process low_dim inputs (direct concat, no projection)
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            features.append(data)
        
        # Concatenate all features
        result = torch.cat(features, dim=-1)
        return result
    
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
