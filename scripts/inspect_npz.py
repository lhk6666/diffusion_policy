import numpy as np
import sys

path = '/media/dragon_llm/3C09549315E08290/vla_dataset_semantic_ex/sample_000000.npz'
data = np.load(path, allow_pickle=True)
print(f"Keys: {list(data.keys())}")
for k in data.keys():
    try:
        print(f"{k}: {data[k].shape}")
    except:
        print(f"{k}: {data[k]}")
