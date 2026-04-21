"""
train_demo.py — Train a GPC on a haplotype dataset with early stopping.

Examples:
    python3 train_demo.py --data 1K_full.txt --output-dir out/1K \\
        --latents 16 --epochs 200 --patience 25 --seed 1

    python3 train_demo.py --data 10K_full.txt --output-dir out/10K \\
        --latents 128 --epochs 2000 --patience 100 --seed 1
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import pyjuice as juice
import pyjuice.nodes.distributions as dists


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=str, default="1K_full.txt",
                   help="Whitespace-separated 0/1 haplotype file (rows=haplotypes, cols=SNPs)")
    p.add_argument("--output-dir", type=str, default="out/1K",
                   help="Directory for model checkpoint, splits, and log")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--latents", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=25,
                   help="Stop if val LL does not improve for this many epochs")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--pseudocount", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def load_and_split(path, val_frac, test_frac, seed):
    arr = np.loadtxt(path, dtype=np.int8, delimiter=" ")
    n = len(arr)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    arr = arr[perm]
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    test = arr[:n_test]
    val = arr[n_test:n_test + n_val]
    train = arr[n_test + n_val:]
    return train, val, test


def epoch_ll(pc, loader, device, backward):
    total = 0.0
    for (batch,) in loader:
        x = batch.to(device)
        lls = pc(x)
        if backward:
            lls.mean().backward()
        total += lls.mean().item()
    return total / len(loader)


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}  |  Seed: {args.seed}")
    print(f"Loading {args.data} ...")
    train_np, val_np, test_np = load_and_split(
        args.data, args.val_frac, args.test_frac, args.seed,
    )
    n_snps = train_np.shape[1]
    print(f"Train: {len(train_np)}  Val: {len(val_np)}  Test: {len(test_np)}  SNPs: {n_snps}")

    # Persist splits so downstream scripts (generate / impute / plots) use the exact same data.
    np.savetxt(out / "train.txt", train_np, fmt="%d")
    np.savetxt(out / "val.txt",   val_np,   fmt="%d")
    np.savetxt(out / "test.txt",  test_np,  fmt="%d")

    train_t = torch.tensor(train_np, dtype=torch.long)
    val_t   = torch.tensor(val_np,   dtype=torch.long)

    train_loader = DataLoader(TensorDataset(train_t), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(val_t),   batch_size=args.batch_size, shuffle=False)

    print(f"Building HCLT (num_latents={args.latents}) ...")
    ns = juice.structures.HCLT(
        train_t.float().to(device),
        num_latents=args.latents,
        input_dist=dists.Categorical(num_cats=2),
    )
    pc = juice.compile(ns).to(device)
    print(f"Parameters: {ns.num_parameters()}")

    # One warm-up pass to record the CUDA graph (if on GPU).
    with torch.cuda.device(pc.device) if device.type == "cuda" else open(os.devnull):
        for (batch,) in train_loader:
            x = batch.to(device)
            lls = pc(x, record_cudagraph=(device.type == "cuda"))
            lls.mean().backward()
            break

    best_val = -np.inf
    best_epoch = 0
    no_improve = 0
    ckpt_path = out / "gpc_best.jpc"
    log_path = out / "train.log"
    cfg_path = out / "config.json"

    with open(cfg_path, "w") as f:
        json.dump({**vars(args), "n_snps": int(n_snps),
                   "n_train": int(len(train_np)),
                   "n_val":   int(len(val_np)),
                   "n_test":  int(len(test_np))}, f, indent=2)

    print(f"Training up to {args.epochs} epochs (patience={args.patience}) ...")
    with open(log_path, "w") as log_f:
        log_f.write("epoch,train_ll,val_ll,train_s,val_s\n")
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            pc.init_param_flows(flows_memory=0.0)
            train_ll = epoch_ll(pc, train_loader, device, backward=True)
            pc.mini_batch_em(step_size=1.0, pseudocount=args.pseudocount)
            t1 = time.time()
            with torch.no_grad():
                val_ll = epoch_ll(pc, val_loader, device, backward=False)
            t2 = time.time()

            log_f.write(f"{epoch},{train_ll:.4f},{val_ll:.4f},{t1-t0:.2f},{t2-t1:.2f}\n")
            log_f.flush()

            improved = val_ll > best_val
            tag = " *" if improved else ""
            print(f"[{epoch:4d}/{args.epochs}] train LL {train_ll:.3f}  "
                  f"val LL {val_ll:.3f}  ({t1-t0:.1f}s train, {t2-t1:.1f}s val){tag}")

            if improved:
                best_val = val_ll
                best_epoch = epoch
                no_improve = 0
                juice.save(str(ckpt_path), pc)
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"Early stopping: no val improvement for {args.patience} epochs.")
                    break

    print(f"\nBest epoch: {best_epoch}  |  Best val LL: {best_val:.4f}")
    print(f"Saved: {ckpt_path}")
    print(f"Log:   {log_path}")


if __name__ == "__main__":
    main()
