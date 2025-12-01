import zarr
import sys
import os
import shutil
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_zarr_v3_to_v2.py <input_v3_path> <output_v2_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print(f"Converting {input_path} (v3) to {output_path} (v2)...")
    
    # Open input (v3)
    # zarr 3.x automatically detects format
    src = zarr.open(input_path, mode='r')
    
    # Create output (v2)
    dest = zarr.open(output_path, mode='w', zarr_format=2)
    
    # Manually copy groups and arrays
    # zarr.copy_all(src, dest) # NotImplemented in zarr 3.x
    
    def copy_group(source_group, dest_group):
        # Zarr v3 Group object might not have .items() directly exposed or it behaves differently
        # Use .members() or iterate keys
        for name in source_group.keys():
            item = source_group[name]
            if isinstance(item, zarr.Group):
                print(f"Creating group {name}...")
                new_group = dest_group.create_group(name)
                copy_group(item, new_group)
            elif isinstance(item, zarr.Array):
                print(f"Copying array {name}...")
                # Read data
                data = item[:]
                # Create array in destination
                # Zarr v3 library requires create_array even when writing v2 format?
                # Or create_dataset requires shape.
                dest_group.create_array(name, shape=data.shape, dtype=data.dtype, chunks=item.chunks)
                dest_group[name][:] = data
                
    copy_group(src, dest)
    
    # Copy non-zarr files (like episode_meta.json)
    src_path = Path(input_path)
    dest_path = Path(output_path)
    
    for item in src_path.iterdir():
        if item.is_file() and item.name not in ['zarr.json', '.zgroup', '.zarray', '.zattrs']:
            print(f"Copying {item.name}...")
            shutil.copy2(item, dest_path / item.name)
            
    print("Conversion complete.")

if __name__ == "__main__":
    main()
