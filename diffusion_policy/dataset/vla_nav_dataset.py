"""
VLA Navigation Dataset for Diffusion Policy

This dataset loads pre-generated trajectories from a self-contained Zarr store.
Images are stored per-sample (not per-timestep) to save memory.
Each timestep has a sample_idx pointing to its corresponding image.

Expected Zarr structure:
- data/
  - state: (N_steps, 2) float32 - normalized positions [0,1]
  - action: (N_steps, 2) float32 - normalized next positions [0,1]
  - agent_pos: (N_steps, 2) float32 - same as state
  - sample_idx: (N_steps,) int64 - index into img array
  - img: (N_samples, H, W, 3) uint8 - RGB images (one per unique sample)
- meta/
  - episode_ends: (E,) int64 - cumulative episode lengths
"""

from typing import Dict, Optional, List, Tuple
import torch
import numpy as np
import copy
import pathlib
import json
import zarr
from torchvision import transforms
from PIL import Image
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class VLANavDataset(BaseImageDataset):
    """
    VLA Navigation Dataset for Diffusion Policy.
    
    Loads trajectories and images from a self-contained Zarr store.
    Images are indexed by sample_idx to save memory.
    
    Supports two sampling modes:
    - sliding_window: Sample from all valid starting positions (default DP behavior)
    - episode: One sample per episode, starting from beginning (like VLA)
    """
    
    def __init__(
        self,
        zarr_path: str,
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
        sample_mode: str = 'sliding_window',  # 'sliding_window' or 'episode'
        cache_images: bool = True,  # Load all images into RAM for faster access
    ):
        super().__init__()
        
        self.zarr_path = pathlib.Path(zarr_path)
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.sequence_length = horizon
        self.sample_mode = sample_mode
        self.cache_images = cache_images
        
        # Image transform (matches SigLIP preprocessing)
        # SigLIP expects: [0, 255] -> [0, 1] -> normalize with mean=0.5, std=0.5 -> [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),  # [0, 255] -> [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [0, 1] -> [-1, 1]
        ])
        
        # Open Zarr store
        self.root = zarr.open(str(self.zarr_path), mode='r')
        
        # Load arrays into memory for fast access
        print(f"Loading dataset from {zarr_path}...")
        self.states = self.root['data']['state'][:]
        self.actions = self.root['data']['action'][:]
        self.agent_pos = self.root['data']['agent_pos'][:]
        self.sample_idx = self.root['data']['sample_idx'][:]
        self.episode_ends = self.root['meta']['episode_ends'][:]
        
        # Cache images in RAM for much faster data loading
        if cache_images:
            print("Caching images in RAM (this may take a while)...")
            self.images = self.root['data']['img'][:]  # Load ALL images into memory
            print(f"  Cached {len(self.images)} images ({self.images.nbytes / 1e9:.2f} GB)")
        else:
            self.images = self.root['data']['img']  # Keep as zarr array for lazy loading
        
        self.n_steps = len(self.states)
        self.n_episodes = len(self.episode_ends)
        self.n_samples = self.images.shape[0]
        
        # Compute episode starts
        self.episode_starts = np.concatenate([[0], self.episode_ends[:-1]])
        
        # Load episode metadata if available
        meta_path = self.zarr_path / 'episode_meta.json'
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None
        
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
    
    def _build_indices(self, episode_mask: np.ndarray):
        """Build list of valid (episode_idx, start_idx) pairs for sampling.
        
        Two modes:
        - sliding_window: All valid starting positions (standard DP)
        - episode: One sample per episode, starting from index 0 (like VLA)
        """
        self.indices: List[Tuple[int, int]] = []
        
        for ep_idx in range(self.n_episodes):
            if not episode_mask[ep_idx]:
                continue
            
            if self.sample_mode == 'episode':
                # Episode mode: one sample per episode, always start from 0
                self.indices.append((ep_idx, 0))
            else:
                # Sliding window mode: all valid starting positions
                ep_start = self.episode_starts[ep_idx]
                ep_end = self.episode_ends[ep_idx]
                ep_len = ep_end - ep_start
                
                # Generate all valid starting positions
                max_start = ep_len - self.sequence_length + self.pad_before + self.pad_after
                
                for i in range(max(1, max_start)):
                    self.indices.append((ep_idx, i))
    
    def get_validation_dataset(self):
        """Get validation dataset with inverted mask."""
        val_set = copy.copy(self)
        val_set.train_mask = ~self.train_mask
        val_set._build_indices(val_set.train_mask)
        return val_set
    
    def get_normalizer(self, mode: str = 'limits', **kwargs) -> LinearNormalizer:
        """Get normalizer for actions and observations."""
        data = {
            'action': self.actions,
            'agent_pos': self.agent_pos
        }
        
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        
        return normalizer
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def _get_sequence(self, ep_idx: int, local_start: int) -> Dict[str, np.ndarray]:
        """Get a sequence from an episode with proper padding.
        
        Optimized: 
        - Uses vectorized indexing for state/action
        - Only loads ONE image per episode (since all frames use same image in our dataset)
        """
        ep_start = self.episode_starts[ep_idx]
        ep_end = self.episode_ends[ep_idx]
        ep_len = ep_end - ep_start
        
        # Compute all local indices at once
        local_indices = np.arange(self.sequence_length) + local_start - self.pad_before
        
        # Clip to valid range and convert to global indices
        local_indices_clipped = np.clip(local_indices, 0, ep_len - 1)
        global_indices = ep_start + local_indices_clipped
        
        # Batch read state/action/agent_pos (already in memory, very fast)
        seq_state = self.states[global_indices]
        seq_action = self.actions[global_indices]
        seq_agent_pos = self.agent_pos[global_indices]
        
        # OPTIMIZATION: Only load ONE image per episode
        # In our VLA dataset, all frames in an episode share the same image (same sample_idx)
        # So we only need to load the first frame's image
        first_sample_id = self.sample_idx[global_indices[0]]
        single_img = self.images[first_sample_id]  # (H, W, 3)
        
        # Get instruction from metadata if available
        instruction = None
        if self.metadata is not None and ep_idx < len(self.metadata):
            instruction = self.metadata[ep_idx].get('instruction', None)
        
        return {
            'state': seq_state,
            'action': seq_action,
            'agent_pos': seq_agent_pos,
            'img': single_img,  # Single image, not repeated 128 times!
            'instruction': instruction,  # Text instruction for VLA
        }
    
    def _sample_to_data(self, sample: Dict) -> Dict:
        """Convert a sample to the expected format.
        
        Note: For VLA task with n_obs_steps=1, we only need ONE observation image,
        not a sequence of images.
        
        Image preprocessing matches SigLIP requirements:
        - Resize to 224x224
        - Normalize to [-1, 1] with mean=0.5, std=0.5
        """
        # Only take first n_obs_steps of agent_pos for observation
        # Action is full horizon
        agent_pos = sample['agent_pos'][:1].astype(np.float32)  # (1, 2) for n_obs_steps=1
        
        # Process single image with SigLIP-compatible transform
        # (H, W, 3) uint8 -> PIL -> transform -> (3, 224, 224) float32 in [-1, 1]
        img_pil = Image.fromarray(sample['img'])
        img = self.transform(img_pil)  # (3, 224, 224) in [-1, 1]
        img = img.unsqueeze(0)  # (1, 3, 224, 224)
        
        data = {
            'obs': {
                'image': img.numpy(),  # Will be converted to tensor in __getitem__
                'agent_pos': agent_pos,
            },
            'action': sample['action'].astype(np.float32)
        }
        
        # Add instruction if available (for VLA text conditioning)
        if sample.get('instruction') is not None:
            data['obs']['text'] = sample['instruction']
        
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, local_start = self.indices[idx]
        sample = self._get_sequence(ep_idx, local_start)
        data = self._sample_to_data(sample)
        
        # Extract text before converting to torch (text is string, not array)
        text = data['obs'].pop('text', None)
        
        # Convert numpy arrays to torch tensors
        torch_data = dict_apply(data, torch.from_numpy)
        
        # Add text back (as string, not tensor)
        if text is not None:
            torch_data['obs']['text'] = text
        
        return torch_data


class VLANavDatasetFromPath(VLANavDataset):
    """VLA Navigation Dataset that loads train or val from separate paths."""
    
    def __init__(
        self,
        train_path: str,
        val_path: Optional[str] = None,
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        seed: int = 42,
        max_train_episodes: Optional[int] = None,
        sample_mode: str = 'sliding_window',  # 'sliding_window' or 'episode'
        cache_images: bool = True,  # Load images into RAM
    ):
        super().__init__(
            zarr_path=train_path,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            seed=seed,
            val_ratio=0.0,
            max_train_episodes=max_train_episodes,
            sample_mode=sample_mode,
            cache_images=cache_images
        )
        
        self.val_path = val_path
        self._val_dataset = None
        self._cache_images = cache_images
    
    def get_validation_dataset(self):
        """Load validation dataset from separate path."""
        if self._val_dataset is not None:
            return self._val_dataset
            
        if self.val_path is None:
            return super().get_validation_dataset()
        
        self._val_dataset = VLANavDataset(
            zarr_path=self.val_path,
            horizon=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            sample_mode=self.sample_mode,
            cache_images=self._cache_images
        )
        
        return self._val_dataset


def test_dataset():
    """Quick test of the dataset."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python vla_nav_dataset.py <zarr_path>")
        return
    
    zarr_path = sys.argv[1]
    print(f"Loading dataset from {zarr_path}...")
    
    dataset = VLANavDataset(zarr_path, horizon=16, val_ratio=0.1)
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Number of episodes: {dataset.n_episodes}")
    print(f"Number of unique samples (images): {dataset.n_samples}")
    print(f"Total steps: {dataset.n_steps}")
    
    sample = dataset[0]
    print("\nSample structure:")
    for key, val in sample.items():
        if isinstance(val, dict):
            print(f"  {key}:")
            for k, v in val.items():
                print(f"    {k}: {v.shape}, {v.dtype}")
        else:
            print(f"  {key}: {val.shape}, {val.dtype}")
    
    normalizer = dataset.get_normalizer()
    print("\nNormalizer created successfully")
    
    val_dataset = dataset.get_validation_dataset()
    print(f"\nValidation dataset length: {len(val_dataset)}")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_dataset()
