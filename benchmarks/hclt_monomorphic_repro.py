"""Minimal reproduction: does HCLT break when many SNPs are invariant?

At N=500 the first 12,287 SNPs of the high-coverage region contain 4,499
monomorphic columns and GPC dies with an illegal memory access inside a Triton
product-layer kernel; at N=4,006 the same M has no monomorphic columns and runs
fine. This sweeps the fraction of invariant columns at a fixed, small M so the
trigger can be isolated in seconds rather than in the hour a real grid point
takes.

A monomorphic column has zero entropy, so its mutual information with every
other variable is zero. The Chow-Liu step maximises total mutual information, so
those variables can be attached anywhere at no cost; the question is what shape
the resulting tree takes and whether the compiled circuit survives it.

    python hclt_monomorphic_repro.py --snps 2000 --n 500
"""

import argparse
import sys
import time
import traceback
from collections import Counter

import numpy as np
import torch

sys.setrecursionlimit(1000000)


def make_data(n, m, frac_mono, seed=0):
    """m columns of correlated binary data, frac_mono of them forced invariant."""
    rng = np.random.default_rng(seed)
    # A simple Markov chain gives neighbouring SNPs real dependence, so the
    # Chow-Liu tree has genuine structure to find among the informative columns.
    x = np.zeros((n, m), dtype=np.int8)
    x[:, 0] = rng.random(n) < 0.5
    for j in range(1, m):
        flip = rng.random(n) < 0.1
        x[:, j] = np.where(flip, 1 - x[:, j - 1], x[:, j - 1])
    n_mono = int(round(frac_mono * m))
    mono_idx = rng.choice(m, size=n_mono, replace=False)
    x[:, mono_idx] = 0
    return x, sorted(mono_idx)


def tree_stats(ns):
    """Depth and fan-in of the compiled node graph, walked iteratively."""
    seen, stack, fanin = set(), [ns], Counter()
    max_ch = 0
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        chs = getattr(cur, "chs", []) or []
        fanin[len(chs)] += 1
        max_ch = max(max_ch, len(chs))
        stack.extend(chs)
    return dict(nodes=len(seen), max_children=max_ch,
                fanin_hist=dict(sorted(fanin.items())[-4:]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snps", type=int, default=2000)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--latents", type=int, default=128)
    p.add_argument("--fracs", type=float, nargs="+",
                   default=[0.0, 0.1, 0.25, 0.37, 0.5])
    args = p.parse_args()

    import pyjuice as juice
    import pyjuice.nodes.distributions as dists

    dev = torch.device("cuda")
    print(f"N={args.n}  M={args.snps}  latents={args.latents}\n")

    for frac in args.fracs:
        d, mono = make_data(args.n, args.snps, frac)
        x = torch.tensor(d, dtype=torch.long, device=dev)
        print(f"--- monomorphic fraction {frac:.2f} ({len(mono)} of {args.snps})",
              flush=True)
        try:
            t0 = time.time()
            ns = juice.structures.HCLT(x.float(), num_latents=args.latents,
                                       input_dist=dists.Categorical(num_cats=2))
            st = tree_stats(ns)
            print(f"    structure {time.time()-t0:5.1f}s  nodes={st['nodes']} "
                  f"max_children={st['max_children']} top_fanin={st['fanin_hist']}",
                  flush=True)
            t0 = time.time()
            pc = juice.compile(ns).to(dev)
            print(f"    compile   {time.time()-t0:5.1f}s  layers={len(pc.inner_layer_groups)}",
                  flush=True)
            lls = pc(x[:64])
            lls.mean().backward()
            torch.cuda.synchronize()
            print(f"    forward+backward OK   LL={lls.mean().item():.2f}", flush=True)
            del pc, ns
            torch.cuda.empty_cache()
        except Exception:
            print("    FAILED:")
            traceback.print_exc()
            break


if __name__ == "__main__":
    main()
