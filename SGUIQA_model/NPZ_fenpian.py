import os, json, glob, math, argparse
import numpy as np
import torch

CONFIG = [
    # 例：890 张 → 1 片（shard_size=1000）
    #{"npz_root": r'F:\image_data\underwater_image_data\NPZ\NPZ', "out_dir": r'F:\image_data\underwater_image_data\NPZ\SAUD_FP', "shard_size": 480},
    #{"npz_root": r'F:\image_data\underwater_image_data\NPZ\UWIQA_NPZ_8_0.8', "out_dir": r'F:\image_data\underwater_image_data\NPZ\UWIQA_NPZ_8_0.8_FP', "shard_size": 480},
    # 例：2400 张 → 5 片
    #{"npz_root": r'F:\image_data\underwater_image_data\NPZ\SAUD_NPZ1', "out_dir": r'F:\image_data\underwater_image_data\NPZ\SAUD_FP1', "shard_size": 480},
    # 例：960 张 → 2 片（每片 ~800）
    {"npz_root": r'F:\image_data\underwater_image_data\NPZ\UID_NPZ1', "out_dir": r'F:\image_data\underwater_image_data\NPZ\UID_FP1', "shard_size": 480},
]
# =====================================

def load_one_npz(npz_path: str):
    """把单个 .npz 读成 dict（张量），不做重算"""
    with np.load(npz_path, allow_pickle=False) as d:
        meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
        segments = d["segments"]
        edges    = d["edges"] if "edges" in d.files else d["edge_index"]

        edge_index = torch.as_tensor(edges, dtype=torch.long)
        if edge_index.ndim == 2 and edge_index.shape[0] == 2:
            edge_index = edge_index.t().contiguous()  # (2,E)->(E,2)

        obj = {
            "segments": torch.as_tensor(segments, dtype=torch.long),
            "edge_index": edge_index,  # (E,2)
            "meta": meta,
        }
        if "pos_idx" in d.files and "pos_ptr" in d.files:
            obj["pos_idx"] = torch.as_tensor(d["pos_idx"], dtype=torch.int32)
            obj["pos_ptr"] = torch.as_tensor(d["pos_ptr"], dtype=torch.int64)
        if "edge_attr" in d.files:
            obj["edge_attr"] = torch.as_tensor(d["edge_attr"], dtype=torch.float32)
    return obj

def build_shards(npz_root: str, shard_out_dir: str, shard_size: int = 800):
    """
    npz_root      : 原始 .npz 根目录
    shard_out_dir : 输出分片目录（会生成 shard_0000.pt 等和 index.json）
    shard_size    : 每片样本数（建议 800~1200；890/960 用1000，2400用800）
    """
    os.makedirs(shard_out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(npz_root, "**/*.npz"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No .npz found under: {npz_root}")

    key2shard = {}
    n = len(paths)
    n_shard = math.ceil(n / shard_size)
    print(f"[Info] Found {n} .npz files under {npz_root}")
    print(f"[Info] Writing {n_shard} shard(s) (≈{shard_size}/shard) to {shard_out_dir}")

    for si in range(n_shard):
        beg, end = si * shard_size, min(n, (si + 1) * shard_size)
        pack = {}
        for p in paths[beg:end]:
            relkey = os.path.relpath(p, npz_root).replace("\\", "/")
            pack[relkey] = load_one_npz(p)
            key2shard[relkey] = si
        shard_path = os.path.join(shard_out_dir, f"shard_{si:04d}.pt")
        torch.save(pack, shard_path)
        print(f"  -> shard {si:04d}: {end-beg} samples  ({shard_path})")

    index_path = os.path.join(shard_out_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"npz_root": os.path.abspath(npz_root), "key2shard": key2shard}, f)
    print(f"[Done] Index saved to: {index_path}\n")

def main():
    for job in CONFIG:
        build_shards(job["npz_root"], job["out_dir"], shard_size=int(job["shard_size"]))

if __name__ == "__main__":
    main()