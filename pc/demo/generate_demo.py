"""
generate_demo.py — Sample artificial haplotypes from a trained GPC.

Example:
    python3 generate_demo.py --run-dir out/1K --num-samples 5008 --seed 1
"""


import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import pyjuice as juice


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=str, default="out/1K",
                   help="Directory produced by train_demo.py (contains gpc_best.jpc)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Override path to the .jpc file (default: <run-dir>/gpc_best.jpc)")
    p.add_argument("--output", type=str, default=None,
                   help="Output .txt path (default: <run-dir>/samples.txt)")
    p.add_argument("--num-samples", type=int, default=5008)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    ckpt = Path(args.checkpoint) if args.checkpoint else run_dir / "gpc_best.jpc"
    out_path = Path(args.output) if args.output else run_dir / "samples.txt"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Loading {ckpt}")

    pc = juice.compile(juice.load(str(ckpt))).to(device)

    batches = []
    remaining = args.num_samples
    pbar = tqdm(total=args.num_samples, desc="Sampling")
    while remaining > 0:
        k = min(args.batch_size, remaining)
        s = juice.queries.sample(pc, num_samples=k).cpu().numpy()
        batches.append(s)
        remaining -= k
        pbar.update(k)
    pbar.close()

    samples = np.vstack(batches).astype(np.int8)
    np.savetxt(out_path, samples, fmt="%d")
    print(f"Saved {samples.shape[0]} x {samples.shape[1]} samples to {out_path}")


if __name__ == "__main__":
    main()
