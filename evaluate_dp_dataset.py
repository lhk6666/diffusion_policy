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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add diffusion_policy to path
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply


# Standard number of waypoints for fair curvature comparison
STANDARD_NUM_WAYPOINTS = 20


def resample_trajectory(traj: np.ndarray, num_points: int) -> np.ndarray:
    """
    Resample trajectory to fixed number of points using linear interpolation.
    This ensures fair comparison of curvature across different methods.
    
    Args:
        traj: Original trajectory [N, 2]
        num_points: Target number of points
    
    Returns:
        Resampled trajectory [num_points, 2]
    """
    if len(traj) < 2:
        return traj
    
    if len(traj) == num_points:
        return traj
    
    # Compute cumulative arc length
    diffs = np.diff(traj, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_length[-1]
    
    if total_length < 1e-8:
        return np.tile(traj[0], (num_points, 1))
    
    # Generate uniform samples along arc length
    target_lengths = np.linspace(0, total_length, num_points)
    
    # Interpolate
    resampled = np.zeros((num_points, 2))
    for i, target_len in enumerate(target_lengths):
        idx = np.searchsorted(cumulative_length, target_len, side='right') - 1
        idx = np.clip(idx, 0, len(traj) - 2)
        
        seg_start_len = cumulative_length[idx]
        seg_len = segment_lengths[idx] if idx < len(segment_lengths) else 1e-8
        
        if seg_len < 1e-8:
            t = 0
        else:
            t = (target_len - seg_start_len) / seg_len
        t = np.clip(t, 0, 1)
        
        resampled[i] = traj[idx] * (1 - t) + traj[idx + 1] * t
    
    return resampled


class TrajectoryMetrics:
    """
    Trajectory evaluation metrics (same as HMRS/scripts/test/metrics.py)
    - FGE: Final Goal Error (Euclidean distance to goal)
    - CR: Collision Rate (1 if any point hits obstacle, 0 otherwise)
    - PLR: Path Length Ratio (pred_length / gt_length)
    - Curv: Curvature (mean absolute angle change between segments)
    
    Note: Curvature is computed on resampled trajectory (STANDARD_NUM_WAYPOINTS points)
    for fair comparison across methods with different waypoint counts.
    """
    def __init__(self, pred_traj: np.ndarray, gt_traj: np.ndarray, 
                 goal_pos: np.ndarray, obstacle_mask: Optional[np.ndarray] = None):
        self.pred_traj = np.array(pred_traj)
        self.gt_traj = np.array(gt_traj)
        self.goal_pos = np.array(goal_pos)
        self.obstacle_mask = obstacle_mask  # (H, W) binary mask, 1=obstacle
        
    def final_goal_error(self) -> float:
        """FGE: Euclidean distance from final position to goal"""
        if len(self.pred_traj) == 0:
            return float('inf')
        return float(np.linalg.norm(self.pred_traj[-1] - self.goal_pos))
    
    def collision_rate(self) -> float:
        """CR: 1.0 if trajectory collides with obstacle, 0.0 otherwise"""
        if self.obstacle_mask is None:
            return 0.0
        H, W = self.obstacle_mask.shape
        for pt in self.pred_traj:
            # Convert normalized [0,1] to pixel coordinates
            cx = int(pt[0] * W)
            cy = int(pt[1] * H)
            # Clamp to valid range
            cx = np.clip(cx, 0, W - 1)
            cy = np.clip(cy, 0, H - 1)
            if self.obstacle_mask[cy, cx] == 1:
                return 1.0
        return 0.0
    
    def path_length_ratio(self) -> float:
        """PLR: pred_path_length / gt_path_length"""
        if len(self.pred_traj) < 2 or len(self.gt_traj) < 2:
            return 1.0
        pred_len = self._compute_path_length(self.pred_traj)
        gt_len = self._compute_path_length(self.gt_traj)
        return float(pred_len / gt_len) if gt_len > 1e-6 else 1.0
    
    def curvature(self, num_points: int = STANDARD_NUM_WAYPOINTS) -> float:
        """Curv: Mean absolute angle change between consecutive segments (radians)
        
        Note: Trajectory is resampled to num_points for fair comparison.
        """
        resampled_traj = resample_trajectory(self.pred_traj, num_points)
        return self._compute_curvature(resampled_traj)
    
    def _compute_path_length(self, path: np.ndarray) -> float:
        """Compute total path length"""
        if len(path) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    
    def _compute_curvature(self, path: np.ndarray) -> float:
        """Compute mean curvature as angle change between segments"""
        if len(path) < 3:
            return 0.0
        vectors = path[1:] - path[:-1]
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
    if 'state_dicts' in payload and model_key in payload['state_dicts']:
        old_state = payload['state_dicts'][model_key]
        new_state = {}
        for k, v in old_state.items():
            # Rename self_attn -> cross_attn in fusion layers (old checkpoint compatibility)
            if '.fusion.layers.' in k and '.self_attn.' in k:
                new_k = k.replace('.self_attn.', '.cross_attn.')
                print(f"  Renaming: {k.split('.')[-2]}.{k.split('.')[-1]} -> cross_attn.*")
                new_state[new_k] = v
            else:
                new_state[k] = v
        payload['state_dicts'][model_key] = new_state
    
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
                       default='/media/dragon_llm/linux_ssd/vla_dp_224/val',
                       help='Path to validation Zarr dataset')
    parser.add_argument('--output_dir', '-o', type=str, default='dp_eval_results',
                       help='Output directory')
    parser.add_argument('--num_episodes', type=int, default=None,
                       help='Number of episodes (None = all)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--use_rollout', action='store_true',
                       help='Use receding horizon rollout instead of single-shot')
    parser.add_argument('--visualize_every', type=int, default=100,
                       help='Visualize every N episodes (0 = disabled)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference (default: 32)')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    policy, cfg = load_dp_model(args.checkpoint, args.device)
    print(f"Model loaded. Horizon: {policy.horizon}, n_action_steps: {policy.n_action_steps}")
    
    # Load dataset
    print(f"\nLoading dataset from {args.dataset}")
    dataset = DPZarrDataset(args.dataset, load_mask=True)
    
    num_episodes = args.num_episodes or dataset.num_episodes
    num_episodes = min(num_episodes, dataset.num_episodes)
    
    print(f"\nEvaluating {num_episodes} episodes with batch_size={args.batch_size}...")
    
    results = []
    all_pred_trajs = []
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
                pred_trajs = predict_trajectory_batch(
                    policy, images, starts, instructions, args.device
                )
            except Exception as e:
                print(f"Batch inference failed: {e}, falling back to single inference")
                pred_trajs = []
                for ep in batch_episodes:
                    try:
                        traj = predict_trajectory(
                            policy, ep.image, ep.start, ep.instruction, args.device
                        )
                        pred_trajs.append(traj)
                    except:
                        pred_trajs.append(np.zeros((10, 2)))
            
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
                
                try:
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
    
    # Per-scene results
    scene_results = {}
    for r in valid_results:
        scene_id = r.get('scene_id', 'unknown')
        if scene_id not in scene_results:
            scene_results[scene_id] = []
        scene_results[scene_id].append(r)
    
    print("\n" + "=" * 60)
    print("PER-SCENE RESULTS")
    print("=" * 60)
    
    scene_summary = []
    for scene_id, scene_res in sorted(scene_results.items()):
        scene_fge = np.mean([r['fge'] for r in scene_res])
        scene_cr = np.mean([r['cr'] for r in scene_res])
        scene_plr = np.mean([r['plr'] for r in scene_res])
        scene_curv = np.mean([r['curv'] for r in scene_res])
        print(f"{scene_id}: FGE={scene_fge:.4f}, CR={scene_cr*100:.1f}%, PLR={scene_plr:.3f}, Curv={scene_curv:.4f}, N={len(scene_res)}")
        scene_summary.append({
            'scene_id': scene_id,
            'fge': float(scene_fge),
            'cr': float(scene_cr),
            'plr': float(scene_plr),
            'curv': float(scene_curv),
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
                'num_valid': len(valid_results),
                'num_with_mask': mask_count
            },
            'per_scene': scene_summary,
            'episode_results': results
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Save CSV
    csv_file = output_dir / "metrics.csv"
    with open(csv_file, 'w') as f:
        f.write("episode_idx,scene_id,instruction,fge,cr,plr,curv\n")
        for r in results:
            instruction = r.get('instruction', '')[:50].replace(',', ' ').replace('\n', ' ')
            f.write(f"{r.get('episode_idx', '')},{r.get('scene_id', '')},{instruction},"
                   f"{r.get('fge', '')},{r.get('cr', '')},{r.get('plr', '')},{r.get('curv', '')}\n")
    
    print(f"CSV saved to {csv_file}")


if __name__ == "__main__":
    main()
