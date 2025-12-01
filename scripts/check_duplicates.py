import json
from collections import Counter

path = '/media/dragon_llm/3C09549315E08290/vla_dataset_semantic_ex_traj_v2/episode_meta.json'
print(f"Loading {path}...")

with open(path, 'r') as f:
    data = json.load(f)

print(f"Total entries: {len(data)}")

sample_ids = [d['sample_id'] for d in data]
unique_ids = set(sample_ids)
print(f"Unique Sample IDs: {len(unique_ids)}")

counts = Counter(sample_ids)
print(f"Most common IDs: {counts.most_common(10)}")
