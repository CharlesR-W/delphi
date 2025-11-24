
import orjson
import sys
from pathlib import Path

path = Path("/home/charles/results/bestofk_baseline/scores/fuzz/multi_scores/layers.5.mlp_latent1_0.txt")
try:
    data = orjson.loads(path.read_bytes())
    n_pos = sum(1 for x in data if x.get("activating"))
    n_neg = sum(1 for x in data if not x.get("activating"))
    print(f"File: {path.name}")
    print(f"Pos: {n_pos}, Neg: {n_neg}, Ratio: {n_pos/(n_pos+n_neg):.2f}")
    print(f"Total: {len(data)}")
except Exception as e:
    print(e)

