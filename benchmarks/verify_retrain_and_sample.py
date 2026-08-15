"""Can a GPC be trained from scratch, saved, reloaded and sampled from?

Models saved with the older PyJuice cannot be sampled under current main: their
pickled `Categorical` objects predate the `sampling_fns` attribute the new
sampler expects. That leaves the question of whether a model trained *under* the
new version works end to end, which is what the public instructions need to be
true.

This walks the whole path a user follows: learn the structure, compile, run EM,
sample, save, reload, sample again. A small model is enough, since the failure
is an API mismatch rather than anything scale-dependent.

    PYTHONPATH=/scratch2/prateek/pyjuice-main/src python verify_retrain_and_sample.py --tag main+fix
"""

import argparse
import os
import sys
import tempfile
import traceback

import numpy as np
import torch

sys.setrecursionlimit(1000000)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN = f"{ROOT}/results/1KG/8020/data/8020_train.txt"
TEST = f"{ROOT}/results/1KG/8020/data/8020_test.txt"


def step(label, fn):
    try:
        out = fn()
        print(f"  [ok]   {label}", flush=True)
        return out
    except Exception:
        print(f"  [FAIL] {label}", flush=True)
        traceback.print_exc()
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--snps", type=int, default=2047)
    p.add_argument("--haplotypes", type=int, default=1000)
    p.add_argument("--latents", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--num-samples", type=int, default=500)
    args = p.parse_args()

    import pyjuice as juice
    import pyjuice.nodes.distributions as dists
    print(f"[{args.tag}] pyjuice from {os.path.dirname(juice.__file__)}", flush=True)

    device = torch.device("cuda")
    tr = np.loadtxt(TRAIN, dtype=np.int8, delimiter=' ')[:args.haplotypes, :args.snps]
    te = np.loadtxt(TEST, dtype=np.int8, delimiter=' ')[:, :args.snps]
    x = torch.tensor(tr, dtype=torch.long)
    print(f"  train {x.shape}, {args.latents} latents, {args.epochs} epochs", flush=True)

    ns = step("structure learning (HCLT)", lambda: juice.structures.HCLT(
        x.float().to(device), num_latents=args.latents,
        input_dist=dists.Categorical(num_cats=2)))
    pc = step("compile", lambda: juice.compile(ns).to(device))

    def train():
        nb = max(1, -(-x.shape[0] // 128))
        splits = np.array_split(np.arange(x.shape[0]), nb)
        for _ in range(args.epochs):
            pc.init_param_flows(flows_memory=0.0)
            for idx in splits:
                pc(x[idx].to(device)).mean().backward()
            pc.mini_batch_em(step_size=1.0, pseudocount=0.005)
        with torch.no_grad():
            t = torch.tensor(te, dtype=torch.long)
            return float(np.mean([pc(t[s:s + 256].to(device)).mean().item()
                                  for s in range(0, t.shape[0], 256)]))
    ll = step("EM training", train)
    print(f"         held-out LL after {args.epochs} epochs: {ll:.3f}", flush=True)

    def sample_from(model, label):
        s = juice.queries.sample(model, num_samples=args.num_samples).cpu().numpy()
        corr = float(np.corrcoef(s.mean(0), te.mean(0))[0, 1])
        print(f"         {label}: {s.shape}, allele-freq corr with real {corr:.4f}", flush=True)
        return s

    step("sample from the freshly trained model",
         lambda: sample_from(pc, "samples"))

    path = os.path.join(tempfile.gettempdir(), f"verify_{args.tag}.jpc")
    step("save", lambda: juice.save(path, pc))
    pc2 = step("reload", lambda: juice.compile(juice.load(path)).to(device))
    step("sample from the reloaded model",
         lambda: sample_from(pc2, "samples after reload"))
    os.remove(path)
    print(f"\n[{args.tag}] full train -> save -> reload -> sample path works", flush=True)


if __name__ == "__main__":
    main()
