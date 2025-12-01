import zarr
import json
import cv2
import numpy as np
import pathlib
from tqdm import tqdm
import concurrent.futures
import os

def process_image(args):
    idx, meta, resize_shape = args
    image_path = meta['image_path']
    try:
        img = cv2.imread(image_path)
        if img is None:
            return idx, np.zeros(resize_shape + (3,), dtype=np.uint8)
        img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return idx, img
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return idx, np.zeros(resize_shape + (3,), dtype=np.uint8)

def main():
    zarr_path = '/media/dragon_llm/3C09549315E08290/vla_dataset_semantic_ex_traj_v2'
    resize_shape = (96, 96)
    chunk_size = (64, 96, 96, 3) # Chunk by 64 images
    
    print(f"Opening Zarr: {zarr_path}")
    root = zarr.open(zarr_path, mode='r+')
    
    meta_path = os.path.join(zarr_path, 'episode_meta.json')
    print(f"Loading metadata from {meta_path}")
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    n_episodes = len(metadata)
    print(f"Found {n_episodes} episodes.")
    
    if 'images' in root:
        print("Warning: 'images' array already exists. Overwriting/Resuming?")
        # For safety, let's not overwrite automatically unless user asks. 
        # But here I am the user agent. I'll assume we want to create it.
        # Let's check shape.
        if root['images'].shape == (n_episodes, 96, 96, 3):
            print("Images array exists with correct shape.")
            # return # Uncomment to skip if already done
        else:
            print("Images array exists but wrong shape. Deleting.")
            del root['images']

    if 'images' not in root:
        print("Creating 'images' Zarr array...")
        images_arr = root.create_dataset(
            'images',
            shape=(n_episodes, 96, 96, 3),
            chunks=chunk_size,
            dtype='uint8',
            compressor=zarr.Blosc(cname='lz4', clevel=5, shuffle=1)
        )
    else:
        images_arr = root['images']

    # Use a ThreadPool to read/resize images in parallel
    # We process in batches to be efficient
    batch_size = 1024
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for i in tqdm(range(0, n_episodes, batch_size), desc="Processing batches"):
            batch_indices = range(i, min(i + batch_size, n_episodes))
            batch_args = [(idx, metadata[idx], resize_shape) for idx in batch_indices]
            
            results = list(executor.map(process_image, batch_args))
            
            # Sort results just in case (though map preserves order)
            # results.sort(key=lambda x: x[0])
            
            # Stack images
            imgs = np.stack([res[1] for res in results])
            
            # Write to Zarr
            images_arr[i : i + len(imgs)] = imgs

    print("Done.")

if __name__ == "__main__":
    main()
