"""Where the Chow-Liu step's memory and time actually go.

`chow_liu_tree` in PyJuice materialises a networkx graph with one edge object per
pair of variables before taking a maximum spanning tree. This measures that path
against the identical spanning tree computed straight from the dense pairwise
matrix with scipy, so the avoidable part of the cost can be separated from the
part inherent to Chow-Liu.

Each size runs in a fresh subprocess and peak memory is read from VmHWM, because
``ru_maxrss`` is a high-water mark for the whole process: reusing it across sizes
in one process silently understates every measurement after the first.

    python chow_liu_cost.py --snps 1000 2000 3000 4000 6000
"""

import argparse
import json
import os
import subprocess
import sys
import time


def vmhwm_gib():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 2**20
    return float("nan")


def child(m, method):
    import numpy as np
    rng = np.random.default_rng(0)
    mi = rng.random((m, m)).astype(np.float32)
    mi = (mi + mi.T) / 2
    np.fill_diagonal(mi, 0.0)
    before = vmhwm_gib()

    t0 = time.perf_counter()
    if method == "networkx":
        sys.path.insert(0, "/scratch2/prateek/pyjuice/src")
        from pyjuice.structures.hclt import chow_liu_tree
        T = chow_liu_tree(mi)
        n_edges = T.number_of_edges()
    else:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import minimum_spanning_tree
        T = minimum_spanning_tree(csr_matrix(-mi))
        n_edges = T.nnz
    elapsed = time.perf_counter() - t0

    print(json.dumps(dict(M=m, method=method, sec=elapsed, peak_gib=vmhwm_gib(),
                          before_gib=before, tree_edges=n_edges,
                          matrix_gib=4 * m * m / 2**30)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snps", type=int, nargs="+", default=[1000, 2000, 3000, 4000, 6000])
    p.add_argument("--child", type=int, default=None)
    p.add_argument("--method", default="networkx")
    args = p.parse_args()

    if args.child is not None:
        child(args.child, args.method)
        return

    rows = []
    print(f"{'M':>6} {'method':>9} {'pairs':>10} {'peak':>9} {'graph only':>11} "
          f"{'B/pair':>8} {'sec':>8}")
    for m in args.snps:
        for method in ("networkx", "scipy"):
            out = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--child", str(m),
                 "--method", method],
                capture_output=True, text=True)
            if out.returncode != 0:
                print(f"{m:>6} {method:>9}  FAILED: {out.stderr.strip()[-90:]}")
                continue
            r = json.loads(out.stdout.strip().splitlines()[-1])
            pairs = m * (m - 1) / 2
            graph = r["peak_gib"] - r["before_gib"]
            rows.append(r | dict(graph_gib=graph, bytes_per_pair=graph * 2**30 / pairs))
            print(f"{m:>6} {method:>9} {pairs/1e6:9.1f}M {r['peak_gib']:8.2f}G "
                  f"{graph:10.2f}G {graph*2**30/pairs:8.0f} {r['sec']:8.1f}")

    # fit exponents on the largest three sizes of each method
    import math
    for method in ("networkx", "scipy"):
        pts = [(r["M"], r["sec"], r["graph_gib"]) for r in rows if r["method"] == method]
        if len(pts) < 2:
            continue
        e_t = [math.log(b[1] / a[1]) / math.log(b[0] / a[0]) for a, b in zip(pts, pts[1:])]
        e_m = [math.log(b[2] / a[2]) / math.log(b[0] / a[0])
               for a, b in zip(pts, pts[1:]) if a[2] > 0 and b[2] > 0]
        print(f"\n{method}: time ~ M^{sum(e_t)/len(e_t):.2f}"
              + (f", memory ~ M^{sum(e_m)/len(e_m):.2f}" if e_m else ""))


if __name__ == "__main__":
    main()
