"""Do the two spanning-tree routes give the same Chow-Liu tree on real data?

`chow_liu_cost.py` shows the dense-matrix route uses about seven times less
memory, but it measures cost on random matrices and never checks that the trees
agree. That gap matters: PyJuice builds a networkx graph over all pairs and takes
a minimum spanning tree of -MI, while the cheap route hands -MI straight to
scipy. Two things can separate them on real data that cannot on random data.

  ties        monomorphic and near-monomorphic SNPs give MI at or near zero for
              many pairs at once, and a maximum spanning tree is only unique when
              the weights are distinct. Different tie-breaking gives a different
              tree.
  dropped 0s  scipy's sparse format treats an exact zero as an absent edge, not a
              zero-weight one, so any MI of exactly zero leaves the graph. If that
              disconnects a SNP the result is a forest, not a spanning tree.

Both trees are optimal for Chow-Liu whenever their total weight matches, since
any maximum spanning tree attains the same objective. So total weight is the
claim worth testing; edge-set identity is the stronger property and is reported
separately.

    python tree_equivalence.py --snps 8191 --haplotypes 4006
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snps", type=int, default=8191)
    p.add_argument("--haplotypes", type=int, default=4006)
    p.add_argument("--num-bins", type=int, default=32)
    p.add_argument("--sigma", type=float, default=0.5 / 32)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--symmetrize", action="store_true",
                   help="mirror the upper triangle so both routes see one graph")
    args = p.parse_args()

    sys.path.insert(0, "/scratch2/prateek/pyjuice/src")
    from pyjuice.structures.hclt import chow_liu_tree, mutual_information_chunked
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

    from scaling_benchmark import load_data

    data = load_data(args.haplotypes, args.snps)
    mono = int((data.std(axis=0) == 0).sum())
    print(f"data {data.shape}, {mono} monomorphic SNPs "
          f"({mono / args.snps * 100:.1f}%)", flush=True)

    x = torch.tensor(data, dtype=torch.float32, device="cuda")
    t0 = time.perf_counter()
    mi = mutual_information_chunked(x, x, args.num_bins, args.sigma,
                                    args.chunk_size).cpu().numpy()
    mi = np.ascontiguousarray(mi.astype(np.float64))
    print(f"MI matrix in {time.perf_counter() - t0:.1f}s  "
          f"min {mi.min():.3e}  max {mi.max():.3e}", flush=True)

    # PyJuice normalises each chunk by its own min/max, so mi[i,j] and mi[j,i]
    # come from different chunks and need not agree. chow_liu_tree reads only the
    # upper triangle; handing scipy the full matrix would let it pick the other
    # direction, which compares two different graphs rather than two routes.
    asym = np.abs(mi - mi.T)
    print(f"asymmetry |mi - mi.T|: max {asym.max():.3e}, "
          f"mean {asym.mean():.3e}, exactly symmetric: {bool((asym == 0).all())}",
          flush=True)
    if args.symmetrize:
        mi = np.triu(mi, 1)
        mi = mi + mi.T
        print("  symmetrised to the upper triangle both routes should see",
              flush=True)

    # how much room is there for the two routes to disagree?
    iu = np.triu_indices(args.snps, k=1)
    off = mi[iu]
    exact_zeros = int((off == 0).sum())
    uniq = len(np.unique(off))
    print(f"off-diagonal pairs {off.size:,}: {exact_zeros:,} exactly zero, "
          f"{off.size - uniq:,} tied values", flush=True)

    t0 = time.perf_counter()
    T_nx = chow_liu_tree(mi)
    nx_sec = time.perf_counter() - t0
    nx_edges = {frozenset((u, v)) for u, v in T_nx.edges()}
    nx_weight = sum(mi[u, v] for u, v in T_nx.edges())

    t0 = time.perf_counter()
    T_sp = minimum_spanning_tree(csr_matrix(-mi))
    sp_sec = time.perf_counter() - t0
    rows, cols = T_sp.nonzero()
    sp_edges = {frozenset((int(u), int(v))) for u, v in zip(rows, cols)}
    sp_weight = sum(mi[u, v] for u, v in zip(rows, cols))

    ncomp = connected_components(T_sp, directed=False)[0]
    print(f"\nnetworkx  {len(nx_edges):,} edges, total MI {nx_weight:.6f}, {nx_sec:.1f}s")
    print(f"scipy     {len(sp_edges):,} edges, total MI {sp_weight:.6f}, {sp_sec:.1f}s")
    print(f"          expected {args.snps - 1:,} edges for a spanning tree; "
          f"scipy graph has {ncomp} connected component(s)")

    shared = nx_edges & sp_edges
    print(f"\nedges in common       {len(shared):,} of {len(nx_edges):,} "
          f"({len(shared) / max(len(nx_edges), 1) * 100:.2f}%)")
    print(f"only in networkx      {len(nx_edges - sp_edges):,}")
    print(f"only in scipy         {len(sp_edges - nx_edges):,}")

    dw = abs(nx_weight - sp_weight)
    rel = dw / abs(nx_weight) if nx_weight else float("nan")
    print(f"\ntotal-weight difference {dw:.3e} (relative {rel:.3e})")
    if nx_edges == sp_edges:
        print("VERDICT: identical trees")
    elif rel < 1e-9:
        print("VERDICT: different edges, identical objective -- both are optimal "
              "Chow-Liu trees, so the substitution is sound")
    else:
        print("VERDICT: the objectives differ; the routes are NOT interchangeable")


if __name__ == "__main__":
    main()
