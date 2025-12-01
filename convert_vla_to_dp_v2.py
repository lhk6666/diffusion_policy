import os
import zarr
import numpy as np
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import cv2
import concurrent.futures

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

def process_single_file(args):
    npz_path, max_steps = args
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
            meta = {}
            
        # Get goal from npz if available, else meta
        if 'goal' in data:
            goal_pos = data['goal']
        else:
            goal_pos = meta.get('goal_pos', [0,0])

        # Generate trajectories
        trajectories = []
        
        # Sample random start positions from valid mask
        valid_y, valid_x = np.where(mask)
        
        if len(valid_x) > 0:
            # Try to find a valid trajectory
            # We want 1 trajectory per scene
            for _ in range(20): # Try 20 random starts
                idx = np.random.randint(len(valid_x))
                start_pos = [valid_x[idx], valid_y[idx]]
                
                # Skip if too close to goal
                if np.linalg.norm(np.array(start_pos) - np.array(goal_pos)) < 20:
                    continue

                traj, success = integrate_trajectory(flow, mask, start_pos, goal_pos, max_steps=max_steps)
                
                if success and len(traj) > 10: # Filter short/failed
                    trajectories.append(traj)
                    break
        
        results = []
        for traj in trajectories:
            # Normalize state to [0,1] (assuming image size is known or in meta)
            # Usually flow is in pixel coordinates.
            # We need image size.
            H, W = mask.shape
            
            # State: (T, 2)
            states = traj / np.array([W, H])
            
            # Action: (T, 2) - Delta position (velocity)
            # We can compute it from trajectory
            actions = np.zeros_like(states)
            actions[:-1] = states[1:] - states[:-1]
            actions[-1] = actions[-2] # Repeat last action
            
            # Metadata
            # We need to store image path.
            # Assuming image is named same as npz but .png
            image_path = str(npz_path.with_suffix('.png'))
            
            # Convert numpy types to python types for JSON serialization
            goal_list = goal_pos.tolist() if isinstance(goal_pos, np.ndarray) else list(goal_pos)
            
            meta_dict = {
                'image_path': meta.get('image_path', image_path),
                'instruction': meta.get('instruction', ''),
                'goal': goal_list,
                'sample_id': int(meta.get('sample_id', 0))
            }
            
            results.append({
                'states': states.astype(np.float32),
                'actions': actions.astype(np.float32),
                'meta': meta_dict
            })
            
        return results

    except Exception as e:
        print(f"Error processing {npz_path}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to VLA dataset (folder with .npz)')
    parser.add_argument('--output_dir', type=str, default='data/vla_nav_zarr', help='Output Zarr path')
    parser.add_argument('--num_episodes_per_scene', type=int, default=1, help='Number of trajectories to generate per scene')
    parser.add_argument('--max_steps', type=int, default=100, help='Max steps for trajectory')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of parallel workers')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    
    if output_dir.exists():
        print(f"Warning: Output directory {output_dir} exists.")
    
    # Find all npz files
    npz_files = sorted(list(data_root.glob("sample_*.npz")))
    print(f"Found {len(npz_files)} samples in {data_root}")
    
    # Storage for Zarr
    episode_metadata = []
    
    all_states = []
    all_actions = []
    all_episode_indices = []
    episode_ends = []
    
    total_steps = 0
    
    print(f"Generating trajectories from Flow Fields using {args.num_workers} workers...")
    
    # Prepare args for workers
    worker_args = [(f, args.max_steps) for f in npz_files]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Use tqdm to show progress
        results_iter = list(tqdm(executor.map(process_single_file, worker_args), total=len(worker_args)))
        
    print("Aggregating results...")
    for file_results in results_iter:
        if not file_results:
            continue
            
        for res in file_results:
            states = res['states']
            actions = res['actions']
            meta = res['meta']
            
            all_states.append(states)
            all_actions.append(actions)
            
            episode_len = len(states)
            total_steps += episode_len
            episode_ends.append(total_steps)
            
            episode_metadata.append(meta)
            
            # Episode index for each step
            all_episode_indices.append(np.full((episode_len, 1), len(episode_metadata)-1, dtype=np.float32))

    # Concatenate
    print("Concatenating data...")
    if len(all_states) == 0:
        print("No valid trajectories generated!")
        return

    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    all_episode_indices = np.concatenate(all_episode_indices, axis=0)
    episode_ends = np.array(episode_ends, dtype=np.int64)
    
    print(f"Total steps: {total_steps}")
    print(f"Total episodes: {len(episode_ends)}")
    
    # Save to Zarr
    print(f"Saving to {output_dir}...")
    # Force Zarr v2 format for compatibility with older environments (robodiff uses zarr 2.12.0)
    root = zarr.open(str(output_dir), mode='w')
    
    data_group = root.create_group('data')
    
    # Use create_dataset for Zarr v2 compatibility
    data_group.create_dataset('state', data=all_states, chunks=(1000, 2))
    data_group.create_dataset('action', data=all_actions, chunks=(1000, 2))
    data_group.create_dataset('episode_index', data=all_episode_indices, chunks=(1000, 1))
    
    meta_group = root.create_group('meta')
    meta_group.create_dataset('episode_ends', data=episode_ends)
    
    # Save metadata json
    with open(output_dir / 'episode_meta.json', 'w') as f:
        json.dump(episode_metadata, f, indent=2)
        
    print("Done.")

if __name__ == "__main__":
    main()
