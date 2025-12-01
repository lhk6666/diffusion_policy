from typing import Dict
import torch
import numpy as np
import copy
import pathlib
import json
import cv2
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class VLADataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            use_cache=True # Enable RAM cache by default
            ):
        
        super().__init__()
        self.zarr_path = pathlib.Path(zarr_path)
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'episode_index'])
        
        # Load metadata
        meta_path = self.zarr_path / 'episode_meta.json'
        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)
            
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        
        self.use_cache = use_cache
        self.image_cache = None
        
        if self.use_cache:
            print(f"Loading {len(self.metadata)} images into RAM cache...")
            # Pre-allocate memory: (N, 96, 96, 3) uint8
            # 140k * 27KB ~= 3.8GB
            self.image_cache = np.zeros((len(self.metadata), 96, 96, 3), dtype=np.uint8)
            
            from tqdm import tqdm
            for i, meta in tqdm(enumerate(self.metadata), total=len(self.metadata), desc="Caching Images"):
                image_path = meta['image_path']
                img = cv2.imread(image_path)
                if img is None:
                    # Black image fallback
                    img = np.zeros((96, 96, 3), dtype=np.uint8)
                else:
                    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.image_cache[i] = img
            print("Cache loaded.")

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:2]
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # sample['state']: (T, 2)
        # sample['action']: (T, 2)
        # sample['episode_index']: (T, 1)
        
        agent_pos = sample['state'].astype(np.float32) 
        
        # Get image
        # Assuming the image is constant for the episode
        episode_idx = int(sample['episode_index'][0,0])
        
        if self.use_cache and self.image_cache is not None:
            img = self.image_cache[episode_idx]
        else:
            meta = self.metadata[episode_idx]
            image_path = meta['image_path']
            
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                img = np.zeros((96, 96, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        # Normalize to [0,1] and move channel to first dim
        img = img.astype(np.float32) / 255.0
        img = np.moveaxis(img, -1, 0) # 3, 96, 96
        
        # Repeat for horizon
        # (T, 3, 96, 96)
        T = sample['state'].shape[0]
        image = np.repeat(img[np.newaxis,...], T, axis=0)

        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
                'agent_pos': agent_pos, # T, 2
            },
            'action': sample['action'].astype(np.float32) # T, 2
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
