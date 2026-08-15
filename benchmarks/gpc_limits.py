"""What actually stops GPC from fitting a larger region, as computable laws.

Four independent costs grow with the number of SNPs M, and whichever hits its
ceiling first is the one that produces an error. This measures each of them on
real data so the ceiling can be stated as a number rather than a guess.

  1. Mutual-information matrix. `HCLT` allocates a dense M x M float32 on the
     host: 4*M^2 bytes.
  2. Chow-Liu graph. `chow_liu_tree` adds an explicit networkx edge for every
     pair before taking a spanning tree: M(M-1)/2 edge objects, each costing a
     few hundred bytes of Python dict. This is the dominant memory term and the
     reason build time grows as roughly M^2.5.
  3. Tree depth. The circuit is built and traversed by recursion, once per node,
     so a deep backbone exhausts the 8 MB C stack exactly as the chain-structured
     HMM does past ~6,000 SNPs. Genomic LD is local, so the maximum spanning tree
     tends to follow the chromosome and can be far deeper than a balanced tree.
  4. GPU memory for the compiled circuit, which is the only term the latent count
     affects.

Terms 1-3 are computed here without ever building the circuit, by taking the
spanning tree with scipy instead of networkx, so a ladder of M can be scanned in
minutes rather than days.

    python gpc_limits.py --snps 2047 4095 8191 12287 14670 20000
"""

import argparse
import csv
import os
import resource
import sys
import time
from collections import deque

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_DATA = f"{ROOT}/results/b38/8020/data/8020_train.txt"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "limits")


def build_matrix(n, m, _cache={}):
    """Tile the real region when m exceeds it, permuting rows per tile."""
    if "d" not in _cache:
        _cache["d"] = np.loadtxt(BASE_DATA, dtype=np.int8, delimiter=' ')
    full = _cache["d"][:n]
    if m <= full.shape[1]:
        return np.ascontiguousarray(full[:, :m])
    rng = np.random.default_rng(0)
    tiles = []
    while sum(t.shape[1] for t in tiles) < m:
        tiles.append(full[rng.permutation(full.shape[0])])
    return np.ascontiguousarray(np.concatenate(tiles, axis=1)[:, :m])


def tree_depth(edges, num_nodes):
    """Depth of the spanning tree from its centre, and its maximum degree."""
    adj = [[] for _ in range(num_nodes)]
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)

    def bfs(src):
        dist = [-1] * num_nodes
        dist[src] = 0
        q = deque([src])
        far = src
        while q:
            u = q.popleft()
            for w in adj[u]:
                if dist[w] < 0:
                    dist[w] = dist[u] + 1
                    far = w if dist[w] > dist[far] else far
                    q.append(w)
        return dist, far

    _, a = bfs(0)          # double sweep gives the diameter endpoints
    dist_a, b = bfs(a)
    diameter = dist_a[b]
    max_degree = max(len(a_) for a_ in adj)
    # rooted at the centre, depth is half the diameter
    return diameter, (diameter + 1) // 2, max_degree


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snps", type=int, nargs="+",
                   default=[2047, 4095, 8191, 12287, 14670, 20000, 29340])
    p.add_argument("--n", type=int, default=4006)
    p.add_argument("--out", default=OUT)
    args = p.parse_args()

    from pyjuice.structures.hclt import mutual_information_chunked
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda")
    rows = []
    print(f"{'M':>7} {'MI mat':>9} {'nx edges':>10} {'MI+MST':>9} "
          f"{'depth':>7} {'diam':>7} {'maxdeg':>7} {'host GiB':>9}")
    for m in args.snps:
        x = torch.tensor(build_matrix(args.n, m), dtype=torch.float, device=device)
        t0 = time.perf_counter()
        mi = mutual_information_chunked(x, x, num_bins=32, sigma=0.5/32,
                                        chunk_size=64).detach().cpu().numpy()
        # maximum spanning tree = minimum spanning tree of the negated weights,
        # the same objective chow_liu_tree optimises, without materialising a
        # networkx edge per pair
        np.fill_diagonal(mi, 0.0)
        mst = minimum_spanning_tree(csr_matrix(-mi))
        elapsed = time.perf_counter() - t0
        coo = mst.tocoo()
        edges = list(zip(coo.row.tolist(), coo.col.tolist()))
        diameter, depth, maxdeg = tree_depth(edges, m)

        row = dict(M=m, N=args.n, mi_matrix_gib=4 * m * m / 2**30,
                   nx_edges=m * (m - 1) // 2,
                   nx_edges_gib=m * (m - 1) // 2 * 200 / 2**30,   # ~200 B/edge
                   mi_mst_sec=elapsed, tree_depth=depth, tree_diameter=diameter,
                   max_degree=maxdeg,
                   host_peak_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
        rows.append(row)
        print(f"{m:>7} {row['mi_matrix_gib']:8.2f}G {row['nx_edges']/1e6:9.0f}M "
              f"{elapsed:8.1f}s {depth:>7} {diameter:>7} {maxdeg:>7} "
              f"{row['host_peak_gib']:8.1f}", flush=True)
        del x, mi, mst
        torch.cuda.empty_cache()

        with open(os.path.join(args.out, "gpc_limits.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    print(f"\nwrote {args.out}/gpc_limits.csv")


if __name__ == "__main__":
    main()
