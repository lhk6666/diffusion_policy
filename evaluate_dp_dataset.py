#!/usr/bin/env python3
"""
Diffusion Policy Evaluation on VLA Dataset

Evaluates trained DP model on validation dataset with the same metrics as VLA:
- FGE (Final Goal Error)
- CR (Collision Rate) 
- PLR (Path Length Ratio)
- Curv (Curvature)

Usage:
python evaluate_dp_dataset.py \
    --checkpoint data/outputs/2025.12.05/.../checkpoints/epoch=0080-val_loss=0.0011.ckpt \
    --output_dir dp_eval_results
"""

import os
import sys
import json
import random
import zarr
import numpy as np
import torch
import hydra
import dill
from pathlib import Path
from tqdm import tqdm
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import re
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Add diffusion_policy to path
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

# Also add FlowVLA root to path so we can share evaluation metrics.
FLOWVLA_ROOT = script_dir.parents[2]
sys.path.insert(0, str(FLOWVLA_ROOT))

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply


# Shared metrics implementation (preferred). Fallback keeps this script runnable standalone.
try:
    from scripts.test.trajectory_metrics import (
        STANDARD_NUM_WAYPOINTS,
        resample_trajectory,
        TrajectoryMetrics,
    )
except Exception:
    STANDARD_NUM_WAYPOINTS = 100

    def resample_trajectory(traj: np.ndarray, num_points: int) -> np.ndarray:
        if traj is None:
            return np.zeros((num_points, 2), dtype=np.float32)
        traj = np.asarray(traj)
        if len(traj) < 2:
            return np.repeat(traj[:1], num_points, axis=0) if len(traj) == 1 else np.zeros((num_points, 2), dtype=np.float32)
        if len(traj) == num_points:
            return traj
        diffs = np.diff(traj, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        cumulative_length = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        total_length = float(cumulative_length[-1])
        if total_length < 1e-8:
            return np.repeat(traj[:1], num_points, axis=0)
        target_lengths = np.linspace(0.0, total_length, num_points)
        resampled = np.zeros((num_points, 2), dtype=traj.dtype)
        seg_idx = 0
        for i, target_len in enumerate(target_lengths):
            while seg_idx < len(segment_lengths) - 1 and cumulative_length[seg_idx + 1] < target_len:
                seg_idx += 1
            seg_start = cumulative_length[seg_idx]
            seg_end = cumulative_length[seg_idx + 1]
            denom = max(seg_end - seg_start, 1e-8)
            alpha = float((target_len - seg_start) / denom)
            resampled[i] = (1.0 - alpha) * traj[seg_idx] + alpha * traj[seg_idx + 1]
        return resampled

    class TrajectoryMetrics:
        def __init__(self, pred_traj: np.ndarray, gt_traj: np.ndarray, goal_pos: np.ndarray, obstacle_mask: Optional[np.ndarray] = None):
            self.pred_traj = np.asarray(pred_traj)
            self.gt_traj = np.asarray(gt_traj)
            self.goal_pos = np.asarray(goal_pos)
            self.obstacle_mask = obstacle_mask

        def final_goal_error(self) -> float:
            if len(self.pred_traj) == 0:
                return float('inf')
            pred_rs = resample_trajectory(self.pred_traj, STANDARD_NUM_WAYPOINTS)
            return float(np.linalg.norm(pred_rs[-1] - self.goal_pos))

        def collision_rate(self) -> float:
            if self.obstacle_mask is None:
                return 0.0
            mask = np.asarray(self.obstacle_mask)
            H, W = mask.shape[:2]
            pred_rs = resample_trajectory(self.pred_traj, STANDARD_NUM_WAYPOINTS)
            for pt in pred_rs:
                cx = int(pt[0] * W)
                cy = int(pt[1] * H)
                cx = np.clip(cx, 0, W - 1)
                cy = np.clip(cy, 0, H - 1)
                if mask[cy, cx] == 1:
                    return 1.0
            return 0.0

        def path_length_ratio(self) -> float:
            if len(self.pred_traj) < 2 or len(self.gt_traj) < 2:
                return 1.0
            pred_rs = resample_trajectory(self.pred_traj, STANDARD_NUM_WAYPOINTS)
            gt_rs = resample_trajectory(self.gt_traj, STANDARD_NUM_WAYPOINTS)
            pred_len = float(np.sum(np.linalg.norm(np.diff(pred_rs, axis=0), axis=1)))
            gt_len = float(np.sum(np.linalg.norm(np.diff(gt_rs, axis=0), axis=1)))
            return float(pred_len / gt_len) if gt_len > 1e-6 else 1.0

        def curvature(self, num_points: int = STANDARD_NUM_WAYPOINTS) -> float:
            resampled_traj = resample_trajectory(self.pred_traj, num_points)
            if len(resampled_traj) < 3:
                return 0.0
            vectors = resampled_traj[1:] - resampled_traj[:-1]
            norms = np.linalg.norm(vectors, axis=1)
            valid = norms > 1e-6
            vectors = vectors[valid]
            if len(vectors) < 2:
                return 0.0
            angles = np.arctan2(vectors[:, 1], vectors[:, 0])
            diffs = angles[1:] - angles[:-1]
            diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
            return float(np.mean(np.abs(diffs)))


@dataclass 
class EpisodeData:
    """Single episode data"""
    episode_idx: int
    sample_id: str
    scene_id: str
    instruction: str
    target_category: str
    direction: str
    goal: np.ndarray           # (2,) normalized [0,1]
    start: np.ndarray          # (2,) normalized [0,1]
    image: np.ndarray          # (H, W, 3) uint8
    gt_trajectory: np.ndarray  # (T, 2) normalized [0,1]
    obstacle_mask: Optional[np.ndarray] = None  # (H, W) binary, 1=obstacle


class DPZarrDataset:
    """Load DP format Zarr dataset.
    
    Supports both:
    1. Legacy format: zarr_path is the zarr root directly
    2. Unified format (generate_vla_v2): zarr_path contains dataset.zarr subdir with embedded mask
    """
    
    def __init__(self, zarr_path: str, load_mask: bool = True):
        self.zarr_path = Path(zarr_path)
        self.load_mask = load_mask
        
        # Handle both direct path and parent directory (unified format)
        if (self.zarr_path / 'dataset.zarr').exists():
            self.zarr_root = self.zarr_path / 'dataset.zarr'
        else:
            self.zarr_root = self.zarr_path
        
        self.root = zarr.open_group(str(self.zarr_root), mode='r')
        
        # Load episode metadata (check both locations)
        episode_meta_path = self.zarr_path / "episode_meta.json"
        if not episode_meta_path.exists():
            episode_meta_path = self.zarr_root.parent / "episode_meta.json"
        with open(episode_meta_path, "r") as f:
            self.episode_meta = json.load(f)
        
        self.images = self.root['data/img']
        self.agent_pos = self.root['data/agent_pos']
        self.episode_ends = self.root['meta/episode_ends'][:]
        
        # Load sample_id to image index mapping
        sample_indices_path = self.zarr_path / 'sample_indices.json'
        if not sample_indices_path.exists():
            sample_indices_path = self.zarr_root.parent / 'sample_indices.json'
        
        if sample_indices_path.exists():
            with open(sample_indices_path, 'r') as f:
                self.sample_id_to_img_idx = json.load(f)
            print(f"  Loaded sample_indices.json: {len(self.sample_id_to_img_idx)} mappings")
        else:
            # Fallback: build from episode_meta
            sample_ids = [m['sample_id'] for m in self.episode_meta]
            unique_sample_ids = sorted(set(sample_ids))
            self.sample_id_to_img_idx = {sid: i for i, sid in enumerate(unique_sample_ids)}
            print(f"  Built sample_id mapping from episode_meta")
        
        # Check for embedded mask (unified format from generate_vla_v2)
        self.has_embedded_mask = 'mask' in self.root['data']
        if self.has_embedded_mask:
            self.masks = self.root['data/mask']
            print(f"  Embedded masks: {self.masks.shape}")
        
        self.num_episodes = len(self.episode_meta)
        print(f"Loaded dataset: {self.num_episodes} episodes, {self.images.shape[0]} images")
    
    def _load_mask(self, sample_id: str, img_idx: int) -> Optional[np.ndarray]:
        """Load obstacle mask.
        
        Returns: (H, W) binary mask where 1=obstacle, 0=free
        """
        if not self.load_mask:
            return None
        
        # Use embedded mask from unified format
        if self.has_embedded_mask:
            try:
                mask_small = np.array(self.masks[img_idx])
                # Convert walkable mask to obstacle mask: 1=obstacle, 0=free
                obstacle_mask = (~mask_small.astype(bool)).astype(np.uint8)
                return obstacle_mask
            except Exception as e:
                print(f"  Warning: Failed to load embedded mask for {sample_id}: {e}")
        
        return None
    
    def get_episode(self, idx: int) -> EpisodeData:
        meta = self.episode_meta[idx]
        start_idx = 0 if idx == 0 else self.episode_ends[idx - 1]
        end_idx = self.episode_ends[idx]
        
        sample_id = meta['sample_id']
        img_idx = self.sample_id_to_img_idx[sample_id]
        image = np.array(self.images[img_idx])
        gt_traj = np.array(self.agent_pos[start_idx:end_idx])
        
        obstacle_mask = self._load_mask(sample_id, img_idx)
        
        return EpisodeData(
            episode_idx=idx,
            sample_id=sample_id,
            scene_id=meta['scene_id'],
            instruction=meta['instruction'],
            target_category=meta['target_category'],
            direction=meta['direction'],
            goal=np.array(meta['goal']),
            start=np.array(meta['start']),
            image=image,
            gt_trajectory=gt_traj,
            obstacle_mask=obstacle_mask
        )


def load_dp_model(checkpoint_path: str, device: str = 'cuda:0'):
    """Load trained Diffusion Policy model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill, map_location=device)
    cfg = payload['cfg']
    
    # Determine which model to use (prefer EMA)
    use_ema = cfg.training.use_ema
    model_key = 'ema_model' if use_ema else 'model'
    
    # Fix state_dict key compatibility: old checkpoints use 'self_attn', new code uses 'cross_attn'
    # Only process the model we need
    # if 'state_dicts' in payload and model_key in payload['state_dicts']:
    #     old_state = payload['state_dicts'][model_key]
    #     new_state = {}
    #     for k, v in old_state.items():
    #         # Rename self_attn -> cross_attn in fusion layers (old checkpoint compatibility)
    #         if '.fusion.layers.' in k and '.self_attn.' in k:
    #             new_k = k.replace('.self_attn.', '.cross_attn.')
    #             print(f"  Renaming: {k.split('.')[-2]}.{k.split('.')[-1]} -> cross_attn.*")
    #             new_state[new_k] = v
    #         else:
    #             new_state[k] = v
    #     payload['state_dicts'][model_key] = new_state
    
    # Create workspace and load only the needed model
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=None)
    
    # Load only the model we need
    exclude_keys = ('model',) if use_ema else ('ema_model',)
    workspace.load_payload(payload, exclude_keys=exclude_keys, include_keys=None)
    
    # Get policy
    policy = workspace.ema_model if use_ema else workspace.model
    
    policy.to(device)
    policy.eval()
    
    print(f"  Using {'EMA' if use_ema else 'regular'} model")
    return policy, cfg


def predict_trajectory(policy, image: np.ndarray, start_pos: np.ndarray, 
                       instruction: str = None,
                       device: str = 'cuda:0', horizon: int = 100) -> np.ndarray:
    """
    Predict trajectory using Diffusion Policy (single sample).
    
    Args:
        policy: DP policy model
        image: (H, W, 3) uint8 image
        start_pos: (2,) normalized start position [0,1]
        instruction: text instruction (e.g., "go to the sofa")
        device: torch device
        horizon: number of steps to predict (rollout)
        
    Returns:
        trajectory: (T, 2) predicted positions
    """
    # Prepare observation
    # DP expects (B, T_obs, C, H, W) for image and (B, T_obs, D) for state
    # Note: normalizer expects 'image' not 'img'
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # (3, H, W)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 3, H, W)
    
    state_tensor = torch.from_numpy(start_pos).float().unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 2)
    
    obs_dict = {
        'image': img_tensor,  # 'image' not 'img' - must match normalizer key
        'agent_pos': state_tensor
    }
    
    # Add text instruction if provided
    if instruction is not None:
        obs_dict['text'] = [instruction]  # Must be a list for batch processing
    # Predict action sequence
    with torch.no_grad():
        result = policy.predict_action(obs_dict)
    
    action = result['action'].cpu().numpy()[0]  # (horizon, 2)
    
    # Build trajectory from start + actions (actions are positions, not deltas)
    # In VLA dataset, action = next position
    trajectory = action  # (horizon, 2)
    
    return trajectory


def predict_trajectory_batch(policy, images: List[np.ndarray], start_positions: List[np.ndarray],
                             instructions: List[str], device: str = 'cuda:0') -> List[np.ndarray]:
    """
    Predict trajectories for a batch of samples (much faster than single inference).
    
    Args:
        policy: DP policy model
        images: List of (H, W, 3) uint8 images
        start_positions: List of (2,) normalized start positions [0,1]
        instructions: List of text instructions
        device: torch device
        
    Returns:
        trajectories: List of (T, 2) predicted positions
    """
    B = len(images)
    if B == 0:
        return []
    
    # Stack images: (B, 1, 3, H, W)
    img_tensors = []
    for img in images:
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = (img_t - 0.5) / 0.5  # Normalize to [-1, 1] if needed
        img_tensors.append(img_t)
    img_batch = torch.stack(img_tensors, dim=0).unsqueeze(1).to(device)  # (B, 1, 3, H, W)
    
    # Stack start positions: (B, 1, 2)
    state_batch = torch.tensor(np.array(start_positions), dtype=torch.float32)
    state_batch = state_batch.unsqueeze(1).to(device)  # (B, 1, 2)
    
    obs_dict = {
        'image': img_batch,
        'agent_pos': state_batch,
        'text': instructions  # List of strings
    }
    
    # Batch predict
    with torch.no_grad():
        result = policy.predict_action(obs_dict)
    
    actions = result['action'].cpu().numpy()  # (B, horizon, 2)
    
    # Split into list
    trajectories = [actions[i] for i in range(B)]
    return trajectories


def rollout_trajectory(policy, image: np.ndarray, start_pos: np.ndarray,
                       goal_pos: np.ndarray, device: str = 'cuda:0',
                       instruction: str = None,
                       max_steps: int = 200, goal_threshold: float = 0.05) -> np.ndarray:
    """
    Rollout trajectory with receding horizon control until reaching goal.
    
    Args:
        policy: DP policy model
        image: (H, W, 3) uint8 image
        start_pos: (2,) start position
        goal_pos: (2,) goal position
        instruction: text instruction
        device: torch device
        max_steps: maximum rollout steps
        goal_threshold: distance to consider goal reached
        
    Returns:
        trajectory: (T, 2) full trajectory
    """
    trajectory = [start_pos.copy()]
    current_pos = start_pos.copy()
    
    n_action_steps = policy.n_action_steps if hasattr(policy, 'n_action_steps') else 8
    
    
    for _ in range(max_steps // n_action_steps):
        # Prepare observation
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
        
        state_tensor = torch.from_numpy(current_pos).float().unsqueeze(0).unsqueeze(0).to(device)
        
        obs_dict = {
            'image': img_tensor,  # 'image' not 'img'
            'agent_pos': state_tensor
        }
        
        # Add text instruction
        if instruction is not None:
            obs_dict['text'] = [instruction]
        
        with torch.no_grad():
            result = policy.predict_action(obs_dict)
        
        actions = result['action'].cpu().numpy()[0]  # (horizon, 2)
        
        # Execute n_action_steps
        for i in range(min(n_action_steps, len(actions))):
            current_pos = actions[i]
            trajectory.append(current_pos.copy())
            
            # Check if goal reached
            if np.linalg.norm(current_pos - goal_pos) < goal_threshold:
                return np.array(trajectory)
    
    return np.array(trajectory)


def evaluate_episode(policy, episode: EpisodeData, device: str = 'cuda:0',
                     use_rollout: bool = False) -> Dict[str, Any]:
    """Evaluate single episode."""
    result = {
        'episode_idx': episode.episode_idx,
        'sample_id': episode.sample_id,
        'scene_id': episode.scene_id,
        'instruction': episode.instruction,
        'target_category': episode.target_category,
        'fge': float('inf'),
        'cr': 0.0,
        'plr': 1.0,
        'curv': 0.0,
        'has_mask': episode.obstacle_mask is not None
    }
    try:
        t0 = time.perf_counter()
        if use_rollout:
            # Use receding horizon rollout
            pred_traj = rollout_trajectory(
                policy, episode.image, episode.start, episode.goal,
                instruction=episode.instruction,  # Pass instruction!
                device=device, max_steps=200
            )
        else:
            # Single-shot prediction
            pred_traj = predict_trajectory(
                policy, episode.image, episode.start,
                instruction=episode.instruction,  # Pass instruction!
                device=device, horizon=len(episode.gt_trajectory)
            )
        t1 = time.perf_counter()
        result['inference_ms'] = float((t1 - t0) * 1000.0)

        # Keep metrics comparable: trim prediction to GT length if needed.
        if pred_traj is not None and len(pred_traj) > len(episode.gt_trajectory):
            pred_traj = pred_traj[:len(episode.gt_trajectory)]
        
        # Compute metrics
        metrics = TrajectoryMetrics(
            pred_traj=pred_traj,
            gt_traj=episode.gt_trajectory,
            goal_pos=episode.goal,
            obstacle_mask=episode.obstacle_mask
        )
        
        result['fge'] = metrics.final_goal_error()
        result['cr'] = metrics.collision_rate()
        result['plr'] = metrics.path_length_ratio()
        result['curv'] = metrics.curvature()
        result['pred_traj_len'] = len(pred_traj)
        result['gt_traj_len'] = len(episode.gt_trajectory)
        
    except Exception as e:
        result['error'] = str(e)
        import traceback
        traceback.print_exc()
    
    return result, pred_traj


def visualize_episode(episode: EpisodeData, pred_traj: np.ndarray, 
                      result: Dict, output_path: Path):
    """Visualize episode prediction vs ground truth."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Show image
    ax.imshow(episode.image)
    
    H, W = episode.image.shape[:2]
    
    # Plot GT trajectory
    gt_pixels = episode.gt_trajectory * np.array([W, H])
    ax.plot(gt_pixels[:, 0], gt_pixels[:, 1], 'g-', linewidth=2, label='GT')
    
    # Plot predicted trajectory  
    pred_pixels = pred_traj * np.array([W, H])
    ax.plot(pred_pixels[:, 0], pred_pixels[:, 1], 'r-', linewidth=2, label='Pred')
    
    # Plot start and goal
    start_px = episode.start * np.array([W, H])
    goal_px = episode.goal * np.array([W, H])
    ax.plot(start_px[0], start_px[1], 'bo', markersize=10, label='Start')
    ax.plot(goal_px[0], goal_px[1], 'r*', markersize=15, label='Goal')
    
    ax.set_title(f"FGE={result['fge']:.3f}, CR={result['cr']:.0f}, PLR={result['plr']:.2f}\n{episode.instruction[:50]}")
    ax.legend()
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Diffusion Policy on VLA Dataset')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to DP checkpoint')
    parser.add_argument('--dataset', type=str,
                       default='/media/dragon_llm/linux_ssd/vla_dataset_unified_static_v10/val',
                       help='Path to validation Zarr dataset')
    parser.add_argument('--output_dir', '-o', type=str, default='dp_eval_results',
                       help='Output directory')
    parser.add_argument('--num_episodes', type=int, default=None,
                       help='Number of episodes (None = all)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--use_rollout', action='store_true',
                       help='Use receding horizon rollout instead of single-shot')
    parser.add_argument('--visualize_every', type=int, default=1000,
                       help='Visualize every N episodes (0 = disabled)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference (default: 32)')
    parser.add_argument('--num_inference_steps', type=int, default=None,
                       help='Override diffusion denoising steps at inference time (if supported by the policy)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Optional random seed for reproducible evaluation')
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(int(args.seed))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    policy, cfg = load_dp_model(args.checkpoint, args.device)

    # Optionally override diffusion sampling iterations.
    # Most DiffusionPolicy implementations expose this as `policy.num_inference_steps`.
    if args.num_inference_steps is not None:
        if hasattr(policy, 'num_inference_steps'):
            policy.num_inference_steps = int(args.num_inference_steps)
        else:
            print("Warning: policy has no attribute 'num_inference_steps'; ignoring --num_inference_steps")
    print(f"Model loaded. Horizon: {policy.horizon}, n_action_steps: {policy.n_action_steps}")
    if hasattr(policy, 'num_inference_steps'):
        print(f"Inference denoising steps: {policy.num_inference_steps}")
    
    # Load dataset
    print(f"\nLoading dataset from {args.dataset}")
    dataset = DPZarrDataset(args.dataset, load_mask=True)
    
    num_episodes = args.num_episodes or dataset.num_episodes
    num_episodes = min(num_episodes, dataset.num_episodes)
    
    print(f"\nEvaluating {num_episodes} episodes with batch_size={args.batch_size}...")
    
    results = []
    all_pred_trajs = []
    inference_ms_values: List[float] = []
    vis_dir = output_dir / "visualizations"
    if args.visualize_every > 0:
        vis_dir.mkdir(exist_ok=True)
    
    # Batch evaluation (much faster!)
    if not args.use_rollout:
        # Collect all episodes first
        all_episodes = [dataset.get_episode(idx) for idx in range(num_episodes)]
        
        # Process in batches
        for batch_start in tqdm(range(0, num_episodes, args.batch_size), desc="Batches"):
            batch_end = min(batch_start + args.batch_size, num_episodes)
            batch_episodes = all_episodes[batch_start:batch_end]
            
            # Prepare batch inputs
            images = [ep.image for ep in batch_episodes]
            starts = [ep.start for ep in batch_episodes]
            instructions = [ep.instruction for ep in batch_episodes]
            
            # Batch inference
            try:
                t0 = time.perf_counter()
                pred_trajs = predict_trajectory_batch(
                    policy, images, starts, instructions, args.device
                )
                t1 = time.perf_counter()
                per_episode_ms = float((t1 - t0) * 1000.0 / max(1, len(batch_episodes)))
                batch_inference_ms = [per_episode_ms for _ in range(len(batch_episodes))]
            except Exception as e:
                print(f"Batch inference failed: {e}, falling back to single inference")
                pred_trajs = []
                batch_inference_ms = []
                for ep in batch_episodes:
                    try:
                        t0 = time.perf_counter()
                        traj = predict_trajectory(
                            policy, ep.image, ep.start, ep.instruction, args.device
                        )
                        t1 = time.perf_counter()
                        pred_trajs.append(traj)
                        batch_inference_ms.append(float((t1 - t0) * 1000.0))
                    except:
                        pred_trajs.append(np.zeros((10, 2)))
                        batch_inference_ms.append(float('nan'))
            
            # Compute metrics for each episode in batch
            for i, (episode, pred_traj) in enumerate(zip(batch_episodes, pred_trajs)):
                idx = batch_start + i
                
                result = {
                    'episode_idx': episode.episode_idx,
                    'sample_id': episode.sample_id,
                    'scene_id': episode.scene_id,
                    'instruction': episode.instruction,
                    'target_category': episode.target_category,
                    'fge': float('inf'),
                    'cr': 0.0,
                    'plr': 1.0,
                    'curv': 0.0,
                    'has_mask': episode.obstacle_mask is not None
                }
                if i < len(batch_inference_ms):
                    result['inference_ms'] = float(batch_inference_ms[i])
                
                try:
                    # Keep metrics comparable: trim prediction to GT length if needed.
                    if pred_traj is not None and len(pred_traj) > len(episode.gt_trajectory):
                        pred_traj = pred_traj[:len(episode.gt_trajectory)]
                    metrics = TrajectoryMetrics(
                        pred_traj=pred_traj,
                        gt_traj=episode.gt_trajectory,
                        goal_pos=episode.goal,
                        obstacle_mask=episode.obstacle_mask
                    )
                    result['fge'] = metrics.final_goal_error()
                    result['cr'] = metrics.collision_rate()
                    result['plr'] = metrics.path_length_ratio()
                    result['curv'] = metrics.curvature()
                    result['pred_traj_len'] = len(pred_traj)
                    result['gt_traj_len'] = len(episode.gt_trajectory)
                except Exception as e:
                    result['error'] = str(e)
                
                results.append(result)
                all_pred_trajs.append(pred_traj)
                if np.isfinite(result.get('inference_ms', float('nan'))):
                    inference_ms_values.append(float(result['inference_ms']))
                
                # Visualize
                if args.visualize_every > 0 and idx % args.visualize_every == 0:
                    try:
                        vis_path = vis_dir / f"episode_{idx:05d}.png"
                        visualize_episode(episode, pred_traj, result, vis_path)
                    except Exception as e:
                        print(f"Visualization failed for episode {idx}: {e}")
    else:
        # Rollout mode: must be sequential (receding horizon)
        for idx in tqdm(range(num_episodes), desc="Rollout"):
            episode = dataset.get_episode(idx)
            result, pred_traj = evaluate_episode(policy, episode, args.device, args.use_rollout)
            results.append(result)
            all_pred_trajs.append(pred_traj)
            if np.isfinite(result.get('inference_ms', float('nan'))):
                inference_ms_values.append(float(result['inference_ms']))
            
            # Visualize
            if args.visualize_every > 0 and idx % args.visualize_every == 0:
                try:
                    vis_path = vis_dir / f"episode_{idx:05d}.png"
                    visualize_episode(episode, pred_traj, result, vis_path)
                except Exception as e:
                    print(f"Visualization failed for episode {idx}: {e}")
    
    # Compute aggregate metrics
    valid_results = [r for r in results if r.get('fge') != float('inf')]
    
    fge_values = [r['fge'] for r in valid_results]
    cr_values = [r['cr'] for r in valid_results]
    plr_values = [r['plr'] for r in valid_results]
    curv_values = [r['curv'] for r in valid_results]
    mask_count = sum(1 for r in valid_results if r.get('has_mask', False))
    valid_inference_ms = [float(r['inference_ms']) for r in valid_results if np.isfinite(r.get('inference_ms', float('nan')))]
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY - Diffusion Policy")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Total episodes: {len(results)}")
    print(f"Valid episodes: {len(valid_results)}")
    print(f"Episodes with mask: {mask_count}/{len(valid_results)}")
    print(f"\nTrajectory Metrics:")
    print(f"  FGE  (Final Goal Error):  {np.mean(fge_values):.4f} ± {np.std(fge_values):.4f}")
    print(f"  CR   (Collision Rate):    {np.mean(cr_values)*100:.2f}%")
    print(f"  PLR  (Path Length Ratio): {np.mean(plr_values):.4f} ± {np.std(plr_values):.4f}")
    print(f"  Curv (Curvature):         {np.mean(curv_values):.4f} ± {np.std(curv_values):.4f}")
    if len(valid_inference_ms) > 0:
        print(f"\nPerformance:")
        print(f"  Inference latency:        {np.mean(valid_inference_ms):.2f} ms/episode ± {np.std(valid_inference_ms):.2f} ms")

    def scene_group_id(scene_id: str) -> str:
        """Group scene variants like scene700_01/scene700_02 into scene700."""
        if not scene_id:
            return 'unknown'
        s = str(scene_id).strip()
        s = s.split('/')[-1].split('\\')[-1]
        m = re.match(r'^(scene\d+)(?:[_-]\d+)?$', s)
        if m:
            return m.group(1)
        parts = re.split(r'[_-]', s)
        if parts and re.fullmatch(r'scene\d+', parts[0]):
            return parts[0]
        m = re.match(r'^(scene\d+)', s)
        if m:
            return m.group(1)
        return s
    
    # Per-scene results
    scene_results = {}
    for r in valid_results:
        scene_id_raw = r.get('scene_id', 'unknown')
        scene_id = scene_group_id(scene_id_raw)
        if scene_id not in scene_results:
            scene_results[scene_id] = []
        scene_results[scene_id].append(r)
    
    print("\n" + "=" * 60)
    print("PER-SCENE RESULTS")
    print("=" * 60)
    
    scene_summary = []
    for scene_id, scene_res in sorted(scene_results.items()):
        scene_fge_vals = [r['fge'] for r in scene_res]
        scene_cr_vals = [r['cr'] for r in scene_res]
        scene_plr_vals = [r['plr'] for r in scene_res]
        scene_curv_vals = [r['curv'] for r in scene_res]

        scene_fge = float(np.mean(scene_fge_vals))
        scene_fge_std = float(np.std(scene_fge_vals))
        scene_cr = float(np.mean(scene_cr_vals))
        scene_cr_std = float(np.std(scene_cr_vals))
        scene_plr = float(np.mean(scene_plr_vals))
        scene_plr_std = float(np.std(scene_plr_vals))
        scene_curv = float(np.mean(scene_curv_vals))
        scene_curv_std = float(np.std(scene_curv_vals))

        print(
            f"{scene_id}: FGE={scene_fge:.4f} ± {scene_fge_std:.4f}, "
            f"CR={scene_cr*100:.1f}% ± {scene_cr_std*100:.1f}%, "
            f"PLR={scene_plr:.3f} ± {scene_plr_std:.3f}, "
            f"Curv={scene_curv:.4f} ± {scene_curv_std:.4f}, N={len(scene_res)}"
        )
        scene_summary.append({
            'scene_id': scene_id,
            'fge': scene_fge,
            'fge_std': scene_fge_std,
            'cr': scene_cr,
            'cr_std': scene_cr_std,
            'plr': scene_plr,
            'plr_std': scene_plr_std,
            'curv': scene_curv,
            'curv_std': scene_curv_std,
            'num_episodes': len(scene_res)
        })
    
    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'checkpoint': args.checkpoint,
                'dataset': args.dataset,
                'num_episodes': len(results),
                'use_rollout': args.use_rollout
            },
            'overall': {
                'fge_mean': float(np.mean(fge_values)),
                'fge_std': float(np.std(fge_values)),
                'cr_mean': float(np.mean(cr_values)),
                'plr_mean': float(np.mean(plr_values)),
                'plr_std': float(np.std(plr_values)),
                'curv_mean': float(np.mean(curv_values)),
                'curv_std': float(np.std(curv_values)),
                'inference_ms_mean': float(np.mean(valid_inference_ms)) if len(valid_inference_ms) > 0 else None,
                'inference_ms_std': float(np.std(valid_inference_ms)) if len(valid_inference_ms) > 0 else None,
                'num_valid': len(valid_results),
                'num_with_mask': mask_count
            },
            'per_scene': scene_summary,
            'episode_results': results
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")

    # Save per-scene CSV (grouped scenes + std)
    scene_csv_file = output_dir / "scene_metrics.csv"
    with open(scene_csv_file, 'w') as f:
        f.write("scene_id,num_episodes,fge,fge_std,cr,cr_std,plr,plr_std,curv,curv_std\n")
        for s in scene_summary:
            f.write(
                f"{s.get('scene_id','')},{s.get('num_episodes','')},"
                f"{s.get('fge','')},{s.get('fge_std','')},"
                f"{s.get('cr','')},{s.get('cr_std','')},"
                f"{s.get('plr','')},{s.get('plr_std','')},"
                f"{s.get('curv','')},{s.get('curv_std','')}\n"
            )
    print(f"Per-scene CSV saved to {scene_csv_file}")
    
    # Save CSV
    csv_file = output_dir / "metrics.csv"
    with open(csv_file, 'w') as f:
        f.write("episode_idx,scene_id,instruction,fge,cr,plr,curv,inference_ms\n")
        for r in results:
            instruction = r.get('instruction', '')[:50].replace(',', ' ').replace('\n', ' ')
            f.write(f"{r.get('episode_idx', '')},{r.get('scene_id', '')},{instruction},"
                   f"{r.get('fge', '')},{r.get('cr', '')},{r.get('plr', '')},{r.get('curv', '')},{r.get('inference_ms', '')}\n")
    
    print(f"CSV saved to {csv_file}")


if __name__ == "__main__":
    main()
