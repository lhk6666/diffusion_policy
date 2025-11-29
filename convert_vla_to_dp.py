import os
import zarr
import numpy as np
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import cv2

def integrate_trajectory(flow, mask, start_pos, goal_pos, step_size=1.0, max_steps=200, dist_thresh=5.0):
    """
    Integrate a trajectory from the flow field.
    start_pos: (x, y)
    flow: (2, H, W)
    """
    H, W = mask.shape
    traj = [start_pos]
    curr_pos = np.array(start_pos, dtype=np.float32)
    
    success = False
    
    for _ in range(max_steps):
        # Check bounds
        cx, cy = int(curr_pos[0]), int(curr_pos[1])
        if cx < 0 or cx >= W or cy < 0 or cy >= H:
            break
            
        # Check collision (optional, if mask is strict)
        if not mask[cy, cx]:
            break
            
        # Check goal
        dist = np.linalg.norm(curr_pos - goal_pos)
        if dist < dist_thresh:
            success = True
            break
            
        # Get velocity
        # Bilinear interpolation could be better, but nearest is fast for generation
        vx = flow[0, cy, cx]
        vy = flow[1, cy, cx]
        
        # Normalize velocity to step size (optional, or keep raw flow magnitude)
        # Assuming flow is already normalized or reasonable
        # If flow is unit vector, we multiply by step_size
        
        # Update
        curr_pos[0] += vx * step_size
        curr_pos[1] += vy * step_size
        
        traj.append(curr_pos.copy())
        
    return np.array(traj), success

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to VLA dataset (folder with .npz)')
    parser.add_argument('--output_dir', type=str, default='data/vla_nav_zarr', help='Output Zarr path')
    parser.add_argument('--num_episodes_per_scene', type=int, default=50, help='Number of trajectories to generate per scene')
    parser.add_argument('--max_steps', type=int, default=100, help='Max steps for trajectory')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    
    if output_dir.exists():
        print(f"Warning: Output directory {output_dir} exists.")
    
    # Find all npz files
    npz_files = sorted(list(data_root.glob("sample_*.npz")))
    print(f"Found {len(npz_files)} samples in {data_root}")
    
    # Storage for Zarr
    # We will store:
    # data/state: (N, 2) - [x, y] normalized to 0-1
    # data/action: (N, 2) - [vx, vy] normalized
    # meta/episode_ends: (M,)
    
    # We also need to store metadata to link back to images and instructions
    # Since Zarr is for dense arrays, we'll save a separate json for this.
    episode_metadata = []
    
    all_states = []
    all_actions = []
    all_episode_indices = []
    episode_ends = []
    
    total_steps = 0
    
    print("Generating trajectories from Flow Fields...")
    
    for npz_path in tqdm(npz_files):
        try:
            data = np.load(npz_path, allow_pickle=True)
            flow = data['flow'] # (2, H, W)
            mask = data['mask'] # (H, W)
            
            # Load metadata from json if exists, or from npz
            json_path = npz_path.with_suffix('.json')
            if json_path.exists():
                with open(json_path, 'r') as f:
                    meta = json.load(f)
            else:
                # Fallback if stored in npz (unlikely based on previous code)
                continue
                
            # Extract info
            H, W = mask.shape
            goal = data['goal'] # (x, y)
            
            # Generate trajectories
            valid_starts = np.argwhere(mask) # (N, 2) -> (y, x)
            if len(valid_starts) == 0:
                continue
                
            count = 0
            attempts = 0
            max_attempts = args.num_episodes_per_scene * 5
            
            while count < args.num_episodes_per_scene and attempts < max_attempts:
                attempts += 1
                
                # Random start
                idx = np.random.randint(len(valid_starts))
                y, x = valid_starts[idx]
                start_pos = np.array([x, y])
                
                # Integrate
                traj, success = integrate_trajectory(flow, mask, start_pos, goal, max_steps=args.max_steps)
                
                if success and len(traj) > 10: # Filter too short
                    # Normalize trajectory to 0-1
                    traj_norm = traj / np.array([W, H])
                    
                    # Calculate actions (velocity)
                    # Action at t is state[t+1] - state[t] (or velocity from flow)
                    # Let's use the actual displacement as action to be consistent with DP
                    actions = traj_norm[1:] - traj_norm[:-1]
                    states = traj_norm[:-1]
                    
                    # Append
                    all_states.append(states)
                    all_actions.append(actions)
                    
                    # Episode index
                    episode_idx = len(episode_ends)
                    all_episode_indices.append(np.full((len(states), 1), episode_idx, dtype=np.float32))
                    
                    total_steps += len(states)
                    episode_ends.append(total_steps)
                    
                    # Save metadata for this episode
                    episode_metadata.append({
                        'image_path': str(meta.get('image_path', '')),
                        'instruction': str(meta.get('instruction', '')),
                        'goal': goal.tolist(),
                        'sample_id': meta.get('sample_id', '')
                    })
                    
                    count += 1
                    
        except Exception as e:
            print(f"Error processing {npz_path}: {e}")
            continue

    # Convert to numpy
    print("Stacking data...")
    all_states = np.concatenate(all_states, axis=0).astype(np.float32)
    all_actions = np.concatenate(all_actions, axis=0).astype(np.float32)
    all_episode_indices = np.concatenate(all_episode_indices, axis=0).astype(np.float32)
    episode_ends = np.array(episode_ends, dtype=np.int64)
    
    print(f"Total steps: {total_steps}")
    print(f"Total episodes: {len(episode_ends)}")
    
    # Save to Zarr
    print(f"Saving to {output_dir}...")
    root = zarr.open(str(output_dir), mode='w')
    
    data_group = root.create_group('data')
    
    # Use create_array instead of create_dataset for Zarr v3 compatibility
    # state
    data_group.create_array('state', shape=all_states.shape, dtype=all_states.dtype, chunks=(1000, 2))
    data_group['state'][:] = all_states
    
    # action
    data_group.create_array('action', shape=all_actions.shape, dtype=all_actions.dtype, chunks=(1000, 2))
    data_group['action'][:] = all_actions
    
    # episode_index
    data_group.create_array('episode_index', shape=all_episode_indices.shape, dtype=all_episode_indices.dtype, chunks=(1000, 1))
    data_group['episode_index'][:] = all_episode_indices
    
    meta_group = root.create_group('meta')
    meta_group.create_array('episode_ends', shape=episode_ends.shape, dtype=episode_ends.dtype)
    meta_group['episode_ends'][:] = episode_ends
    
    # Save metadata json
    with open(output_dir / 'episode_meta.json', 'w') as f:
        json.dump(episode_metadata, f, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    main()
