"""
evaluate.py — Quality & privacy evaluation for a trained GPC.

Writes into <run-dir>/quality/ (pca, ld_decay, ld_error, clt_tree, clt_summary)
and <run-dir>/privacy/ (aats).

Notes:
  - PCA is fit jointly on Train + Test + GPC, then each is projected into the
    shared PC space and overlaid.
  - LD is computed on SNPs unfixed in every dataset (Train, Test, GPC) so all
    datasets contribute the same set of pairs to each bp bin.

Example:
    python3 evaluate.py --run-dir out/1K --legend 1K_full.legend --seed 1
"""

import argparse
from collections import deque
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic, sem
from sklearn.decomposition import PCA


plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

COLORS = {"Train": "#5A5A5A", "Test": "black", "GPC": "blue"}

FONTSIZE_LABEL  = 18
FONTSIZE_TICK   = 15
FONTSIZE_LEGEND = 14
ALPHA_HL        = 1.0
ALPHA_OTHER     = 0.55
LW_HL           = 3.0
LW_OTHER        = 2.0
MARKERSIZE_HL   = 8
MARKERSIZE_OTHER = 6
PDF_KW = dict(dpi=300, bbox_inches="tight")


def load_haplotypes(path):
    return np.loadtxt(path, dtype=np.int8, delimiter=" ")


def load_positions(legend_path):
    """Legend files have header `id position a0 a1` (space-separated)."""
    df = pd.read_csv(legend_path, sep=" ")
    return df["position"].values.astype(np.int64)


def maybe_subsample(arr, n, rng):
    if len(arr) <= n:
        return arr
    idx = rng.choice(len(arr), size=n, replace=False)
    return arr[idx]


def is_fixed_mask(data):
    # True at SNPs that are monomorphic in `data` (allele count 0 or N).
    n = len(data)
    ac = data.sum(axis=0)
    return (ac == 0) | (ac == n)


def _apply_big_text(ax):
    ax.set_xlabel(ax.get_xlabel(), fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(ax.get_ylabel(), fontsize=FONTSIZE_LABEL)
    ax.tick_params(axis="both", labelsize=FONTSIZE_TICK)


def _style_legend(ax, lw=3.0):
    leg = ax.legend(frameon=False, fontsize=FONTSIZE_LEGEND)
    handles = getattr(leg, "legendHandles", None) or getattr(leg, "legend_handles", [])
    for h in handles:
        try:
            h.set_linewidth(lw)
        except Exception:
            pass


def plot_pca(train, test, samples, out_path):
    # Fit one PCA on Train + Test + GPC, then project each into the shared space.
    pca = PCA(n_components=6)
    combined = np.vstack([train, test, samples]).astype(np.float32)
    pcs = pca.fit_transform(combined)
    n_tr, n_te = len(train), len(test)
    pcs_tr  = pcs[:n_tr]
    pcs_te  = pcs[n_tr:n_tr + n_te]
    pcs_syn = pcs[n_tr + n_te:]

    pairs = [(0, 1), (2, 3), (4, 5)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    for ax, (i, j) in zip(axes, pairs):
        ax.scatter(pcs_syn[:, i], pcs_syn[:, j],
                   s=40, alpha=0.45, color=COLORS["GPC"],
                   label="GPC", rasterized=True, linewidths=0, zorder=1)
        ax.scatter(pcs_tr[:, i], pcs_tr[:, j],
                   s=22, alpha=0.45, color=COLORS["Train"],
                   label="Train", rasterized=True, linewidths=0, zorder=2)
        ax.scatter(pcs_te[:, i], pcs_te[:, j],
                   s=28, alpha=0.60, color=COLORS["Test"],
                   label="Test", rasterized=True, linewidths=0, zorder=3)
        ax.set_xlabel(f"PC{i+1}")
        ax.set_ylabel(f"PC{j+1}")
        _apply_big_text(ax)
    _style_legend(axes[0])
    fig.savefig(out_path, **PDF_KW)
    plt.close(fig)
    print(f"Saved {out_path}")


def compute_ld_r2(geno):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.corrcoef(geno.astype(np.float32), rowvar=False) ** 2


def flatten_upper(mat):
    iu = np.triu_indices(mat.shape[0], k=1)
    return mat[iu]


def pairwise_bp_distance(positions):
    positions = np.asarray(positions, dtype=np.int64)
    diff = np.abs(positions[:, None] - positions[None, :])
    iu = np.triu_indices(len(positions), k=1)
    return diff[iu]


def bin_ld(dists, ld, bin_edges):
    """Bin LD pairs with *fixed* edges so every dataset shares identical bins."""
    valid = ~np.isnan(ld) & (dists >= bin_edges[0])
    d, l = dists[valid], ld[valid]
    mean = binned_statistic(d, l, statistic="mean", bins=bin_edges).statistic
    s    = binned_statistic(d, l, statistic=sem,   bins=bin_edges).statistic
    n    = binned_statistic(d, l, statistic="count", bins=bin_edges).statistic.astype(int)
    return pd.DataFrame({"bin_left": bin_edges[:-1],
                         "LD_mean":  mean,
                         "LD_sem":   s,
                         "n_pairs":  n})


def plot_ld(train, test, samples, positions, out_decay, out_error, nbins=25):
    # Keep SNPs unfixed in every dataset so Test and GPC share identical pair sets.
    fx_tr = is_fixed_mask(train)
    fx_te = is_fixed_mask(test)
    fx_gp = is_fixed_mask(samples)
    kept  = ~(fx_tr | fx_te | fx_gp)

    print(f"  LD SNP filter:")
    print(f"    fixed in Train : {fx_tr.sum()}")
    print(f"    fixed in Test  : {fx_te.sum()}")
    print(f"    fixed in GPC   : {fx_gp.sum()}")
    print(f"    fixed in ≥1    : {(~kept).sum()}")
    print(f"    unfixed in ALL : {kept.sum()}  / {len(kept)} SNPs  (used for LD)")

    if kept.sum() < 2:
        raise SystemExit("Too few SNPs unfixed across all datasets — cannot plot LD.")

    test_k    = test[:,    kept]
    samples_k = samples[:, kept]
    pos_k     = np.asarray(positions)[kept]

    dists = pairwise_bp_distance(pos_k)
    d_min = max(1, int(dists[dists > 0].min()))
    d_max = int(dists.max())
    bin_edges = np.logspace(np.log10(d_min), np.log10(d_max), nbins + 1)

    binned = {}
    for label, data in [("Test", test_k), ("GPC", samples_k)]:
        ld = flatten_upper(compute_ld_r2(data))
        binned[label] = bin_ld(dists, ld, bin_edges)

    # ── decay ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    for label in ["Test", "GPC"]:
        df = binned[label]
        is_hl = (label == "GPC")
        ax.errorbar(df.bin_left, df.LD_mean, df.LD_sem,
                    label=label,
                    alpha=ALPHA_HL if is_hl else ALPHA_OTHER,
                    linewidth=LW_HL if is_hl else LW_OTHER,
                    color=COLORS[label],
                    marker="o" if is_hl else "s",
                    markersize=MARKERSIZE_HL if is_hl else MARKERSIZE_OTHER,
                    markeredgecolor="white", markeredgewidth=0.5,
                    capsize=3, zorder=10 if is_hl else 2)
    ax.set_xscale("log")
    ax.set_xlabel("Distance between SNPs (bp)")
    ax.set_ylabel(r"Average LD ($r^2$)")
    _apply_big_text(ax)
    _style_legend(ax)
    fig.tight_layout()
    fig.savefig(out_decay, **PDF_KW)
    plt.close(fig)
    print(f"Saved {out_decay}")

    # ── signed LD diff (GPC - Test) with propagated SEM ──────────────────────
    te, sy = binned["Test"], binned["GPC"]
    assert np.allclose(te.bin_left.values, sy.bin_left.values), "bin mismatch"
    diff = sy.LD_mean.values - te.LD_mean.values
    err  = np.sqrt(np.nan_to_num(sy.LD_sem.values) ** 2
                   + np.nan_to_num(te.LD_sem.values) ** 2)
    ok = ~np.isnan(diff)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5, zorder=1)
    ax.errorbar(sy.bin_left.values[ok], diff[ok], yerr=err[ok],
                label="GPC",
                alpha=ALPHA_HL, linewidth=LW_HL,
                color=COLORS["GPC"], marker="o",
                markersize=MARKERSIZE_HL,
                markeredgecolor="white", markeredgewidth=0.5,
                capsize=3, zorder=10)
    ax.set_xscale("log")
    ax.set_xlabel("Distance between SNPs (bp)")
    ax.set_ylabel("LD difference")
    _apply_big_text(ax)
    _style_legend(ax)
    fig.tight_layout()
    fig.savefig(out_error, **PDF_KW)
    plt.close(fig)
    print(f"Saved {out_error}")


def mutual_information_chunked(x, num_bins=32, sigma=0.5 / 32, chunk=64):
    device = x.device
    K = x.size(1)
    out = torch.zeros(K, K, device=device)
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-8)
    bins = torch.linspace(0, 1, num_bins, device=device)

    def mi_block(a, b):
        B = a.size(0)
        K1, K2 = a.size(1), b.size(1)
        ap = torch.exp(-0.5 * (a.unsqueeze(2) - bins.view(1, 1, -1)) ** 2 / sigma ** 2)
        bp = torch.exp(-0.5 * (b.unsqueeze(2) - bins.view(1, 1, -1)) ** 2 / sigma ** 2)
        abp = torch.einsum(
            "bia,baj->ij",
            ap.reshape(B, K1 * num_bins, 1),
            bp.reshape(B, 1, K2 * num_bins),
        ).reshape(K1, num_bins, K2, num_bins) / B
        apn = (ap / ap.sum(2, keepdim=True)).mean(0)
        bpn = (bp / bp.sum(2, keepdim=True)).mean(0)
        abpn = abp / abp.sum(dim=(1, 3), keepdim=True)
        m1 = -(apn * torch.log(apn + 1e-4)).sum(1)
        m2 = -(bpn * torch.log(bpn + 1e-4)).sum(1)
        m12 = -(abpn * torch.log(abpn + 1e-4)).sum(dim=(1, 3))
        return m1.unsqueeze(1) + m2.unsqueeze(0) - m12

    for xs in range(0, K, chunk):
        for ys in range(0, K, chunk):
            xe, ye = min(xs + chunk, K), min(ys + chunk, K)
            out[xs:xe, ys:ye] = mi_block(x_n[:, xs:xe], x_n[:, ys:ye])
    return out.cpu().numpy()


def chow_liu_tree(mi):
    K = mi.shape[0]
    G = nx.Graph()
    for v in range(K):
        G.add_node(v)
        for u in range(v):
            G.add_edge(u, v, weight=-mi[u, v])
    return nx.minimum_spanning_tree(G)


def radial_layout(tree, root):
    parent = {root: None}
    order = [root]
    q = deque([root])
    while q:
        v = q.popleft()
        for u in tree.neighbors(v):
            if u not in parent:
                parent[u] = v
                order.append(u)
                q.append(u)
    sub = {v: 1 for v in order}
    for v in reversed(order):
        if parent[v] is not None:
            sub[parent[v]] += sub[v]
    pos = {root: np.array([0.0, 0.0])}
    rng = {root: (0.0, 2 * np.pi)}
    depth = {root: 1}
    q = deque([root])
    while q:
        v = q.popleft()
        a0, a1 = rng[v]
        r = depth[v]
        children = [u for u in tree.neighbors(v) if parent.get(u) == v]
        cursor = a0
        for u in children:
            frac = sub[u] / sub[v]
            a_end = cursor + frac * (a1 - a0)
            mid = 0.5 * (cursor + a_end)
            depth[u] = r + 1
            pos[u] = np.array([r * np.cos(mid), r * np.sin(mid)])
            rng[u] = (cursor, a_end)
            cursor = a_end
            q.append(u)
    return pos


def clt_summary_lines(T):
    K = T.number_of_nodes()
    degrees = dict(T.degree())
    deg = np.array(list(degrees.values()))
    edge_gaps = np.array([abs(u - v) for u, v in T.edges()])
    max_deg_node = max(degrees, key=degrees.get)
    lines = [
        "=" * 60,
        "CLT SUMMARY",
        "=" * 60,
        f"  SNPs (nodes)                   : {K}",
        f"  Edges                          : {T.number_of_edges()}",
        f"  Degree — max                   : {deg.max()}   (node {max_deg_node})",
        f"  Degree — mean                  : {deg.mean():.3f}",
        f"  Degree — median                : {np.median(deg):.1f}",
        f"  Leaf nodes (deg=1)             : {(deg == 1).sum()}   [chain has exactly 2]",
        f"  Degree-2 nodes (chain-like)    : {(deg == 2).sum()}   [chain has K-2 = {K-2}]",
        f"  Hub nodes (deg >= 5)           : {(deg >= 5).sum()}",
        "",
        f"  Edge index gaps — max          : {edge_gaps.max()}",
        f"  Edge index gaps — mean         : {edge_gaps.mean():.2f}",
        f"  Edges with gap > K/10          : {(edge_gaps > K/10).sum()} "
        f"/ {len(edge_gaps)}  ({100*(edge_gaps > K/10).mean():.1f}%)",
        f"  Edges with gap == 1 (adjacent) : {(edge_gaps == 1).sum()}"
        f"   [ALL edges in an HMM]",
        "=" * 60,
    ]
    return lines


def plot_tree(train, out_pdf, out_summary, device):
    print("Computing mutual information ...")
    x = torch.from_numpy(train.astype(np.float32)).to(device)
    mi = mutual_information_chunked(x)
    print("Building Chow-Liu tree ...")
    T = chow_liu_tree(mi)
    K = T.number_of_nodes()
    degrees = dict(T.degree())
    hub_root = max(degrees, key=degrees.get)

    summary = clt_summary_lines(T)
    print("\n" + "\n".join(summary))
    with open(out_summary, "w") as f:
        f.write("\n".join(summary) + "\n")
    print(f"Saved {out_summary}")

    pos = radial_layout(T, hub_root)
    gaps = np.array([abs(u - v) for u, v in T.edges()])
    norm = LogNorm(vmin=1, vmax=max(gaps.max(), 2))
    cmap = matplotlib.colormaps["plasma"]
    order = np.argsort(gaps)
    edges = list(T.edges())
    segs = [(pos[edges[i][0]], pos[edges[i][1]]) for i in order]
    colors = cmap(norm(gaps[order]))

    xy = np.array([pos[n] for n in range(K)])
    deg_arr = np.array([degrees[n] for n in range(K)])
    hub = deg_arr >= 3
    low = ~hub

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_collection(LineCollection(segs, linewidths=0.7, colors=colors, alpha=0.85, zorder=1))
    ax.scatter(xy[low, 0], xy[low, 1], s=3, c="#c8c8c8",
               linewidths=0, alpha=0.6, zorder=2)
    ax.scatter(xy[hub, 0], xy[hub, 1],
               s=np.clip(np.sqrt(deg_arr[hub]) * 8, 10, 120),
               c=deg_arr[hub], cmap="Blues",
               vmin=3, vmax=max(deg_arr.max(), 4),
               linewidths=0.3, edgecolors="#555555", zorder=3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal",
                      shrink=0.5, pad=0.02, aspect=25)
    cb.set_label("SNP distance (log scale)", fontsize=14)
    cb.ax.tick_params(labelsize=12)

    top_hubs = sorted(degrees, key=degrees.get, reverse=True)[:6]
    label_bbox = dict(boxstyle="round,pad=0.3", fc="white", ec="#aaaaaa",
                      lw=0.7, alpha=0.9)
    for node in top_hubs:
        x0, y0 = pos[node]
        r = np.hypot(x0, y0)
        lbl = f"SNP {node}\ndeg {degrees[node]}"
        if r < 1e-9:
            ax.annotate(lbl, xy=(x0, y0), xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=9, fontweight="bold", color="#111",
                        ha="left", va="bottom", bbox=label_bbox)
        else:
            dx, dy = x0 / r, y0 / r
            push = max(r * 0.10, 0.6)
            ax.text(x0 + dx * push, y0 + dy * push, lbl,
                    fontsize=8.5, ha="center", va="center",
                    color="#111", fontweight="bold", bbox=label_bbox)

    ax.autoscale()
    fig.tight_layout()
    fig.savefig(out_pdf, **PDF_KW)
    plt.close(fig)
    print(f"Saved {out_pdf}")


def _l2_matrix(A, B):
    return torch.cdist(A.float(), B.float(), p=2)


def compute_aats(real, syn, device):
    r = torch.from_numpy(real.astype(np.float32)).to(device)
    s = torch.from_numpy(syn.astype(np.float32)).to(device)

    dTT = _l2_matrix(r, r)
    dTT.fill_diagonal_(float("inf"))
    dTT = dTT.min(dim=1).values

    dTS = _l2_matrix(r, s)
    dST = dTS.min(dim=0).values
    dTS = dTS.min(dim=1).values

    dSS = _l2_matrix(s, s)
    dSS.fill_diagonal_(float("inf"))
    dSS = dSS.min(dim=1).values

    aa_truth = (dTS > dTT).float().mean().item()
    aa_syn   = (dST > dSS).float().mean().item()
    return aa_truth, aa_syn


def plot_aats(train, test, samples, out_path, device, rng):
    n = min(len(train), len(test), len(samples) // 2)
    r_tr = maybe_subsample(train, n, rng)
    r_te = maybe_subsample(test, n, rng)
    syn_perm = rng.permutation(len(samples))
    syn_tr = samples[syn_perm[:n]]
    syn_te = samples[syn_perm[n:2 * n]] if len(syn_perm) >= 2 * n else samples[syn_perm[:n]]

    at_tr, as_tr = compute_aats(r_tr, syn_tr, device)
    at_te, as_te = compute_aats(r_te, syn_te, device)

    labels = [r"$AA_{\mathrm{TRUTH}}$" + "\n(Train)",
              r"$AA_{\mathrm{TRUTH}}$" + "\n(Test)",
              r"$AA_{\mathrm{SYN}}$"   + "\n(Train)",
              r"$AA_{\mathrm{SYN}}$"   + "\n(Test)"]
    devs = [abs(at_tr - 0.5), abs(at_te - 0.5), abs(as_tr - 0.5), abs(as_te - 0.5)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(np.arange(4), devs, color=COLORS["GPC"],
                  width=0.55, zorder=3, linewidth=0)
    for b, v in zip(bars, devs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                ha="center", va="bottom", fontsize=11, color="#333")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels, fontsize=FONTSIZE_TICK)
    ax.set_ylim(0, 0.58)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.tick_params(axis="y", labelsize=FONTSIZE_TICK)
    ax.set_ylabel(r"Absolute Error ($|AA - 0.5|$)", fontsize=FONTSIZE_LABEL)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, **PDF_KW)
    plt.close(fig)
    print(f"Saved {out_path}  |  AA_TRUTH (tr/te)={at_tr:.3f}/{at_te:.3f}  "
          f"AA_SYN (tr/te)={as_tr:.3f}/{as_te:.3f}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=str, default="out/1K")
    p.add_argument("--legend", type=str, default="1K_full.legend",
                   help="SNP legend file with `id position a0 a1` header. "
                        "Pass --legend '' to fall back to SNP-index distance.")
    p.add_argument("--samples", type=str, default=None,
                   help="Path to samples.txt (default: <run-dir>/samples.txt)")
    p.add_argument("--skip", type=str, nargs="*", default=[],
                   choices=["pca", "ld", "tree", "aats"],
                   help="Skip specific plots")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    samples_path = Path(args.samples) if args.samples else run_dir / "samples.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    quality_dir = run_dir / "quality"
    privacy_dir = run_dir / "privacy"
    quality_dir.mkdir(parents=True, exist_ok=True)
    privacy_dir.mkdir(parents=True, exist_ok=True)

    print("Loading splits and samples ...")
    train = load_haplotypes(run_dir / "train.txt")
    test = load_haplotypes(run_dir / "test.txt")
    samples = load_haplotypes(samples_path)
    print(f"  train {train.shape}  test {test.shape}  samples {samples.shape}")

    positions = None
    if args.legend:
        positions = load_positions(args.legend)
        if len(positions) != train.shape[1]:
            raise SystemExit(
                f"Legend has {len(positions)} SNPs but data has "
                f"{train.shape[1]} columns — mismatched."
            )
        print(f"  legend: {len(positions)} positions "
              f"({positions.min()} .. {positions.max()} bp)")

    if "pca" not in args.skip:
        plot_pca(train, test, samples, quality_dir / "pca.pdf")
    if "ld" not in args.skip:
        if positions is None:
            positions = np.arange(train.shape[1])
        plot_ld(train, test, samples, positions,
                quality_dir / "ld_decay.pdf", quality_dir / "ld_error.pdf")
    if "tree" not in args.skip:
        plot_tree(train, quality_dir / "clt_tree.pdf",
                  quality_dir / "clt_summary.txt", device)
    if "aats" not in args.skip:
        plot_aats(train, test, samples, privacy_dir / "aats.pdf", device, rng)

    print(f"\nQuality figures -> {quality_dir}")
    print(f"Privacy figures -> {privacy_dir}")


if __name__ == "__main__":
    main()
