"""
VLA Observation Encoder Adapter for Diffusion Policy

This module wraps the VLA Encoder components to work with Diffusion Policy's interface.
It directly reuses the VLA encoder components (SigLIP, ConditionEncoder, CrossModalFusion)
to ensure consistency with the VLA model.

NOTE: This module avoids importing VLAEncoder directly to prevent Lightning dependency
in the robodiff environment. Instead, it recreates the same architecture using nn.Module.

Architecture:
  SigLIPBackbone + ConditionEncoder + CrossModalFusion
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

# Add HMRS to path for imports
hmrs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
if hmrs_path not in sys.path:
    sys.path.insert(0, hmrs_path)

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

# Import VLA encoder COMPONENTS (not VLAEncoder class to avoid Lightning dependency)
from src.models.encoder.siglip_backbone import SigLIPBackbone
from src.models.encoder.condition_encoder import ConditionEncoder
from src.models.encoder.cross_modal_fusion import CrossModalFusion


class VLAObsEncoder(ModuleAttrMixin):
    """
    Adapter that recreates VLAEncoder architecture for Diffusion Policy.
    
    Uses the EXACT SAME components as VLAEncoder:
    - SigLIPBackbone
    - ConditionEncoder  
    - CrossModalFusion
    
    But inherits from nn.Module (via ModuleAttrMixin) instead of LightningModule
    to work in the robodiff environment.
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
        # Recreate VLAEncoder architecture using components
        # This is the EXACT SAME as VLAEncoder but without Lightning
        # ============================================
        
        # 1. SigLIP Backbone
        self.siglip = SigLIPBackbone(
            model_name=siglip_model_name,
            unfreeze_last_n_layers=unfreeze_last_n_layers
        )
        siglip_dim = self.siglip.vision_dim
        
        # 2. Condition Encoder
        self.condition_encoder = ConditionEncoder(
            d_model=d_model,
            siglip_dim=siglip_dim
        )
        
        # 3. Cross-Modal Fusion
        self.fusion = CrossModalFusion(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_fusion_layers,
            d_ff=d_model * 4
        )
        
        # Optional: project low_dim features and concatenate
        low_dim_size = sum(self.key_shape_map[k][-1] for k in self.low_dim_keys)
        if low_dim_size > 0:
            self.low_dim_proj = nn.Linear(low_dim_size, d_model)
            self.output_dim = d_model + d_model  # context + low_dim
        else:
            self.low_dim_proj = None
            self.output_dim = d_model
    
    def encode(self, images: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """
        Encode images and texts using VLA encoder pipeline.
        
        This is the EXACT SAME forward pass as VLAEncoder.forward()
        
        Args:
            images: [B, C, H, W] RGB images
            texts: List[str] text prompts
            
        Returns:
            context: [B, N_v, d_model] fused context tokens
        """
        # 1. SigLIP encode
        image_features = self.siglip.encode_image(images)  # [B, N_v, siglip_dim]
        text_features = self.siglip.encode_text(texts)      # [B, N_t, siglip_dim]
        
        # 2. Project to d_model
        visual_tokens, text_tokens = self.condition_encoder(
            image_features, text_features
        )
        
        # 3. Cross-modal fusion
        context = self.fusion(visual_tokens, text_tokens)  # [B, N_v, d_model]
        
        return context
    
    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode observations for Diffusion Policy.
        
        Args:
            obs_dict: Dictionary with:
                - 'image': [B, C, H, W] RGB image
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
            
            # Normalize to [0, 1] if needed
            if img.max() > 1.0:
                img = img / 255.0
            
            # Get text - use default if not provided
            texts = obs_dict.get('text', None)
            if texts is None:
                texts = [self.default_text] * batch_size
            
            # Forward through encoder pipeline (same as VLAEncoder)
            context = self.encode(img, texts)  # [B, N_v, d_model]
            
            # Pool over visual tokens to get [B, d_model]
            context_pooled = context.mean(dim=1)
            features.append(context_pooled)
        
        # Process low_dim inputs
        if self.low_dim_proj is not None:
            low_dim_features = []
            for key in self.low_dim_keys:
                data = obs_dict[key]
                if batch_size is None:
                    batch_size = data.shape[0]
                low_dim_features.append(data)
            
            if low_dim_features:
                low_dim = torch.cat(low_dim_features, dim=-1)
                low_dim_proj = self.low_dim_proj(low_dim)
                features.append(low_dim_proj)
        
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
        
        # Map VLAEncoder weights to our components
        siglip_dict = {}
        encoder_dict = {}
        fusion_dict = {}
        
        for key, value in state_dict.items():
            if key.startswith('encoder.siglip.'):
                new_key = key[len('encoder.siglip.'):]
                siglip_dict[new_key] = value
            elif key.startswith('encoder.encoder.'):
                new_key = key[len('encoder.encoder.'):]
                encoder_dict[new_key] = value
            elif key.startswith('encoder.fusion.'):
                new_key = key[len('encoder.fusion.'):]
                fusion_dict[new_key] = value
        
        # Load into components
        if siglip_dict:
            self.siglip.load_state_dict(siglip_dict, strict=False)
            print(f"  Loaded {len(siglip_dict)} SigLIP parameters")
        if encoder_dict:
            self.condition_encoder.load_state_dict(encoder_dict, strict=False)
            print(f"  Loaded {len(encoder_dict)} ConditionEncoder parameters")
        if fusion_dict:
            self.fusion.load_state_dict(fusion_dict, strict=False)
            print(f"  Loaded {len(fusion_dict)} CrossModalFusion parameters")


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
    print(f"Components: SigLIP + ConditionEncoder + CrossModalFusion")
    
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
    print("   Uses SAME components as VLAEncoder (SigLIP + ConditionEncoder + Fusion)")


if __name__ == "__main__":
    test_encoder()
