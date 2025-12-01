import zarr
import json
import os

zarr_path = '/media/dragon_llm/3C09549315E08290/vla_dataset_semantic_ex_traj_v2'
root = zarr.open(zarr_path, mode='r')

print("Zarr tree:")
print(root.tree())

if 'data' in root:
    for key in root['data']:
        print(f"data/{key}: {root['data'][key].shape}")

if 'meta' in root:
    if 'episode_ends' in root['meta']:
        print(f"meta/episode_ends: {root['meta']['episode_ends'].shape}")
        n_episodes = root['meta']['episode_ends'].shape[0]
        print(f"Number of episodes: {n_episodes}")

meta_path = os.path.join(zarr_path, 'episode_meta.json')
if os.path.exists(meta_path):
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    print(f"Metadata length: {len(meta)}")
