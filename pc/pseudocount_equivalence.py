"""Check that the batch-size-scaled pseudocount reproduces the bs=256 reference.

Training accumulates flows with ``pc(x).mean().backward()`` once per batch and
applies a single ``mini_batch_em`` per epoch, so every sample enters the update
with weight 1/(its batch size). With near-equal batches the accumulated flow is
therefore (1/eff_batch) times the raw per-sample flow sum, i.e. proportional to
the number of batches rather than to the number of samples. Scaling the
pseudocount by REF_BATCH/eff_batch multiplies it by the same constant, and the
EM update

    theta_i = (flow_i + ps) / sum_j (flow_j + ps)

is invariant to a common rescaling of flow and ps. This script verifies that
empirically by training the same chunk twice and comparing LL trajectories.

    python pseudocount_equivalence.py --snps 1000 --epochs 30
"""

import argparse
import math
import sys

import numpy as np
import torch
import pyjuice as juice

sys.setrecursionlimit(100000)

REF_BATCH = 256
PS = 0.005


def train(data, valid, batch_size, mode, epochs, latents, seed, device):
    """Return the per-epoch (train LL, valid LL) trajectory."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n, seq_length = data.shape
    nb = max(1, math.ceil(n / batch_size))
    eff_batch = n / nb
    ps = PS * (REF_BATCH / eff_batch) if mode == "scaled" else PS

    ns = juice.structures.HMM(seq_length=seq_length, num_latents=latents,
                              homogeneous=False, num_emits=2)
    pc = juice.compile(ns).to(device)

    tr = data.to(device=device, dtype=torch.uint8)
    va = valid.to(device=device, dtype=torch.uint8)
    bounds = [(i * n) // nb for i in range(nb + 1)]
    print(f"  batch {batch_size:5d} -> {nb:3d} batches of ~{eff_batch:.1f}, "
          f"pseudocount {ps:.6g}", flush=True)

    traj = []
    for _ in range(epochs):
        pc.init_param_flows(flows_memory=0.0)
        # Fixed order: the single per-epoch EM update sums flows over all
        # batches, so the permutation cannot matter, and holding it fixed keeps
        # the two runs bit-comparable.
        acc = torch.zeros((), device=device)
        for i in range(nb):
            m = pc(tr[bounds[i]:bounds[i + 1]].long()).mean()
            m.backward()
            acc += m.detach()
        pc.mini_batch_em(step_size=1.0, pseudocount=ps)

        with torch.no_grad():
            vacc = torch.zeros((), device=device)
            nvb = (va.shape[0] + REF_BATCH - 1) // REF_BATCH
            for s in range(0, va.shape[0], REF_BATCH):
                vacc += pc(va[s:s + REF_BATCH].long()).mean()
        traj.append(((acc / nb).item(), (vacc / nvb).item()))

    del pc
    torch.cuda.empty_cache()
    return traj


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default="results/1KG/8020/data/8020_train.txt")
    p.add_argument("--valid", default="results/1KG/8020/data/8020_test.txt")
    p.add_argument("--snps", type=int, default=1000, help="leading SNPs to use")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--latents", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batches", type=int, nargs="+", default=[512, 1024, 2048],
                   help="batch sizes to compare against the bs=256 reference")
    args = p.parse_args()

    device = torch.device("cuda")
    tr = torch.from_numpy(np.loadtxt(args.train, dtype=np.int8, delimiter=' ')[:, :args.snps])
    va = torch.from_numpy(np.loadtxt(args.valid, dtype=np.int8, delimiter=' ')[:, :args.snps])
    print(f"train {tuple(tr.shape)}  valid {tuple(va.shape)}  "
          f"{args.epochs} epochs, {args.latents} latents\n")

    print("reference: batch 256, fixed pseudocount 0.005")
    ref = train(tr, va, REF_BATCH, "fixed", args.epochs, args.latents, args.seed, device)

    print(f"\n{'setting':>28}  {'train LL':>12} {'valid LL':>12}  "
          f"{'|d train|':>10} {'|d valid|':>10}")
    print(f"{'batch 256, ps 0.005 (ref)':>28}  {ref[-1][0]:12.5f} {ref[-1][1]:12.5f}"
          f"  {'--':>10} {'--':>10}")

    for bs in args.batches:
        for mode in ("scaled", "fixed"):
            print(f"\nbatch {bs}, {mode} pseudocount")
            t = train(tr, va, bs, mode, args.epochs, args.latents, args.seed, device)
            dtr = max(abs(a[0] - b[0]) for a, b in zip(t, ref))
            dva = max(abs(a[1] - b[1]) for a, b in zip(t, ref))
            label = f"batch {bs}, {mode}"
            print(f"{label:>28}  {t[-1][0]:12.5f} {t[-1][1]:12.5f}"
                  f"  {dtr:10.2e} {dva:10.2e}   (max over epochs)")


if __name__ == "__main__":
    main()
