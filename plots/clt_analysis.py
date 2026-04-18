"""
analyze_clt.py

Analyzes the Chow-Liu Tree (CLT) backbone of an HCLT model over observed SNPs.

Outputs (all high-quality PDFs, no titles):
  clt_degree_distribution.pdf   — grouped bar chart: CLT vs HMM by degree category
  clt_edge_span.pdf             — distribution of |i-j| for each CLT edge (9999 edges)
  clt_tree.pdf                  — radial tree visualization, nodes colored by degree
"""

from __future__ import annotations

import torch
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter, deque
from matplotlib.collections import LineCollection
import os

np.random.seed(1)
torch.manual_seed(1)

# ── shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        12,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "pdf.fonttype":     42,     # embed fonts properly in PDF
    "ps.fonttype":      42,
})
CLT_COLOR  = "steelblue"
HMM_COLOR  = "tomato"
PDF_KWARGS = dict(format="pdf", bbox_inches="tight", dpi=300)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CORE FUNCTIONS  (from pyjuice/structures/hclt.py)
# ─────────────────────────────────────────────────────────────────────────────

def mutual_information(x1, x2, num_bins, sigma):
    device = x1.device
    B, K1  = x1.size();  K2 = x2.size(1)
    x1 = (x1 - x1.min()) / (x1.max() - x1.min() + 1e-8)
    x2 = (x2 - x2.min()) / (x2.max() - x2.min() + 1e-8)
    bins = torch.linspace(0, 1, num_bins, device=device)
    x1p  = torch.exp(-0.5 * (x1.unsqueeze(2) - bins.view(1, 1, -1)).pow(2) / sigma**2)
    x2p  = torch.exp(-0.5 * (x2.unsqueeze(2) - bins.view(1, 1, -1)).pow(2) / sigma**2)
    x12p = torch.einsum(
        "bia,baj->ij",
        x1p.reshape(B, K1 * num_bins, 1),
        x2p.reshape(B, 1, K2 * num_bins)
    ).reshape(K1, num_bins, K2, num_bins) / B
    x1p_n  = (x1p  / x1p.sum(2,  keepdim=True)).mean(0)
    x2p_n  = (x2p  / x2p.sum(2,  keepdim=True)).mean(0)
    x12p_n = x12p  / x12p.sum(dim=(1, 3), keepdim=True)
    m1  = -(x1p_n  * torch.log(x1p_n  + 1e-4)).sum(1)
    m2  = -(x2p_n  * torch.log(x2p_n  + 1e-4)).sum(1)
    m12 = -(x12p_n * torch.log(x12p_n + 1e-4)).sum(dim=(1, 3))
    return m1.unsqueeze(1) + m2.unsqueeze(0) - m12


def mutual_information_chunked(x1, x2, num_bins, sigma, chunk_size):
    K  = x1.size(1);  mi = torch.zeros([K, K])
    for xs in range(0, K, chunk_size):
        xe = min(xs + chunk_size, K)
        for ys in range(0, K, chunk_size):
            ye = min(ys + chunk_size, K)
            mi[xs:xe, ys:ye] = mutual_information(
                x1[:, xs:xe], x2[:, ys:ye], num_bins, sigma)
    return mi


def chow_liu_tree(mi):
    K = mi.shape[0];  G = nx.Graph()
    for v in range(K):
        G.add_node(v)
        for u in range(v):
            G.add_edge(u, v, weight=-mi[u, v])
    return nx.minimum_spanning_tree(G)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

HAPLOTYPE_FILE = "/scratch2/prateek/genetic_pc_github/results/1KG/8020/data/8020_train.txt"
LEGEND_FILE    = "/scratch2/prateek/genetic_pc_github/aux/10K_SNP.legend"

USE_DUMMY = not (os.path.exists(HAPLOTYPE_FILE) and os.path.exists(LEGEND_FILE))

if USE_DUMMY:
    print("Real data files not found — using synthetic binary haplotype data.")
    N_SAMPLES, N_SNPS = 500, 200
    x      = torch.from_numpy(np.random.randint(0, 2, (N_SAMPLES, N_SNPS)).astype(np.float32))
    pos_bp = np.linspace(27_000_000, 37_000_000, N_SNPS).astype(float)
    print(f"Synthetic data: {N_SAMPLES} haplotypes  x  {N_SNPS} SNPs")
    print(f"Synthetic positions: {pos_bp[0]:.0f} bp  to  {pos_bp[-1]:.0f} bp")
else:
    print(f"Loading haplotype data from:\n  {HAPLOTYPE_FILE}")
    mat = np.loadtxt(HAPLOTYPE_FILE, dtype=np.float32)
    x   = torch.from_numpy(mat)
    assert set(np.unique(mat)).issubset({0.0, 1.0}), \
        "Unexpected values — expected only 0s and 1s."
    print(f"Loaded: {x.shape[0]} haplotypes  x  {x.shape[1]} SNPs")

    print(f"Loading positions from:\n  {LEGEND_FILE}")
    legend = pd.read_csv(LEGEND_FILE, sep=" ")
    pos_bp = legend["position"].values.astype(float)
    assert len(pos_bp) == x.shape[1], (
        f"Legend has {len(pos_bp)} SNPs but haplotype file has {x.shape[1]} columns.")
    print(f"Positions: {pos_bp[0]:.0f} bp  to  {pos_bp[-1]:.0f} bp")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  BUILD THE CLT
# ─────────────────────────────────────────────────────────────────────────────

NUM_BINS   = 32
SIGMA      = 0.5 / 32
CHUNK_SIZE = 64
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nDevice: {DEVICE}")
x = x.to(DEVICE)
print("Computing pairwise mutual information ...")
mi = mutual_information_chunked(x, x, NUM_BINS, SIGMA, CHUNK_SIZE).detach().cpu().numpy()

print("Building Chow-Liu Tree ...")
T    = chow_liu_tree(mi)
K    = T.number_of_nodes()
root = nx.center(T)[0]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

degrees  = dict(T.degree())
deg_vals = np.array(list(degrees.values()))

print("\n" + "=" * 60)
print("CLT SUMMARY")
print("=" * 60)
print(f"  SNPs (nodes)                   : {K}")
print(f"  Edges                          : {T.number_of_edges()}")
print(f"  Degree — max                   : {deg_vals.max()}  "
      f"(node {max(degrees, key=degrees.get)})")
print(f"  Degree — mean                  : {deg_vals.mean():.3f}")
print(f"  Degree — median                : {np.median(deg_vals):.1f}")
print(f"  Leaf nodes (deg=1)             : {(deg_vals == 1).sum()}"
      f"   [chain has exactly 2]")
print(f"  Degree-2 nodes (chain-like)    : {(deg_vals == 2).sum()}"
      f"   [chain has K-2 = {K-2}]")
print(f"  Hub nodes (deg >= 5)           : {(deg_vals >= 5).sum()}")

edge_gaps   = np.array([abs(u - v) for u, v in T.edges()])
edge_bpdist = np.array([abs(pos_bp[u] - pos_bp[v]) for u, v in T.edges()])

print(f"\nEdge index gaps:")
print(f"  Max gap                        : {edge_gaps.max()}")
print(f"  Mean gap                       : {edge_gaps.mean():.1f}")
print(f"  Edges with gap > K/10          : {(edge_gaps > K/10).sum()} "
      f"/ {len(edge_gaps)}  ({100*(edge_gaps > K/10).mean():.1f}%)")
print(f"  Edges with gap == 1 (adjacent) : {(edge_gaps == 1).sum()}"
      f"   [ALL edges in an HMM]")
print(f"\nEdge physical distances:")
print(f"  Max    : {edge_bpdist.max()/1e6:.3f} Mb")
print(f"  Mean   : {edge_bpdist.mean()/1e6:.3f} Mb")
print(f"  Median : {np.median(edge_bpdist)/1e6:.3f} Mb")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PLOT 1 — degree distribution
#     Grouped bar chart: five degree categories on x-axis, CLT vs HMM bars.
#     Categories: 1, 2, 3, 4, ≥5.
# ─────────────────────────────────────────────────────────────────────────────

clt_counts = [int((deg_vals == d).sum()) for d in [1, 2, 3, 4]] + \
             [int((deg_vals >= 5).sum())]
hmm_counts = [2, K - 2, 0, 0, 0]   # chain: exactly 2 leaves, rest deg-2

tick_labels = ["1", "2", "3", "4", "≥5"]
x_pos = np.arange(len(tick_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(x_pos - width/2, clt_counts, width, color=CLT_COLOR,
       label="CLT", edgecolor="white", linewidth=0.5)
ax.bar(x_pos + width/2, hmm_counts, width, color=HMM_COLOR,
       label="HMM", edgecolor="white", linewidth=0.5, alpha=0.85)

ax.set_xticks(x_pos)
ax.set_xticklabels(tick_labels)
ax.set_xlabel("Degree")
ax.set_ylabel("Number of SNPs")
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig("clt_degree_distribution.pdf", **PDF_KWARGS)
plt.close(fig)
print("\nSaved: clt_degree_distribution.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PLOT 2 — edge span distribution
#     X axis: |i-j| for each CLT edge (sequential index gap).
#     An HMM would be a single spike at gap=1.
#     The shape of this distribution directly answers "how non-chain-like is
#     the CLT?" — a heavy tail means many long-range LD connections.
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.hist(edge_gaps, bins=60, color=CLT_COLOR, edgecolor="white",
        linewidth=0.4, label="CLT")
ax.axvline(1, color=HMM_COLOR, linestyle="--", linewidth=1.5, label="HMM")
ax.set_yscale("log")
ax.set_xlabel("SNP distance between connected nodes")
ax.set_ylabel("Number of edges")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("clt_edge_span.pdf", **PDF_KWARGS)
plt.close(fig)
print("Saved: clt_edge_span.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  PLOT 3 — radial tree visualization
#
#     BFS radial layout rooted at the highest-degree hub.
#     Edges colored by log(|i-j|) span: light=local LD, vivid=long-range LD.
#     Drawn short-to-long so long-range edges appear on top.
#     Hub nodes (deg ≥ 3) sized by sqrt(degree), darker fill.
#     Top hubs labeled with their degree.
# ─────────────────────────────────────────────────────────────────────────────

def radial_tree_layout(tree, root):
    """Standard subtree-proportional BFS radial layout.  Returns {node: np.array([x, y])}."""
    parent = {root: None}
    order  = [root]
    queue  = deque([root])
    while queue:
        v = queue.popleft()
        for u in tree.neighbors(v):
            if u not in parent:
                parent[u] = v
                order.append(u)
                queue.append(u)

    subtree_size = {v: 1 for v in order}
    for v in reversed(order):
        p = parent[v]
        if p is not None:
            subtree_size[p] += subtree_size[v]

    pos         = {root: np.array([0.0, 0.0])}
    angle_range = {root: (0.0, 2 * np.pi)}
    depth       = {root: 1}
    queue = deque([root])
    while queue:
        v = queue.popleft()
        a0, a1   = angle_range[v]
        r        = depth[v]
        children = [u for u in tree.neighbors(v) if parent.get(u) == v]
        cursor   = a0
        for u in children:
            frac           = subtree_size[u] / subtree_size[v]
            a_end          = cursor + frac * (a1 - a0)
            mid            = 0.5 * (cursor + a_end)
            depth[u]       = r + 1
            pos[u]         = np.array([r * np.cos(mid), r * np.sin(mid)])
            angle_range[u] = (cursor, a_end)
            cursor         = a_end
            queue.append(u)
    return pos

def power_compress(pos_dict, gamma=0.5):
    """Apply r^gamma to each node's radius from the origin, preserving angles.
    gamma < 1 compresses deeper/wider edges while keeping the tree's shape."""
    result = {}
    for n, (x, y) in pos_dict.items():
        r = np.hypot(x, y)
        if r > 1e-10:
            theta = np.arctan2(y, x)
            result[n] = (r ** gamma) * np.array([np.cos(theta), np.sin(theta)])
        else:
            result[n] = np.array([x, y])
    return result

hub_root = max(degrees, key=degrees.get)
print(f"\nBuilding radial layout rooted at node {hub_root} (deg={degrees[hub_root]}) ...")

pos = radial_tree_layout(T, hub_root)

xy      = np.array([pos[n] for n in range(K)])
deg_arr = np.array([degrees[n] for n in range(K)])

# ── edges: sort short→long so long-range draw on top ──────────────────────
from matplotlib.colors import LogNorm
all_gaps = np.array([abs(u - v) for u, v in T.edges()])
edge_norm = LogNorm(vmin=1, vmax=all_gaps.max())
edge_cmap = matplotlib.colormaps["plasma"]

edge_order = np.argsort(all_gaps)
edge_list  = list(T.edges())
segs_sorted = [(pos[edge_list[i][0]], pos[edge_list[i][1]]) for i in edge_order]
gaps_sorted = all_gaps[edge_order]

# uniform linewidth and alpha — color alone encodes span so all edges are equally visible
colors = edge_cmap(edge_norm(gaps_sorted))

lc = LineCollection(segs_sorted, linewidths=0.55, colors=colors, alpha=0.75, zorder=1)

# ── nodes ─────────────────────────────────────────────────────────────────
mask_low = deg_arr <= 2
mask_hub = deg_arr >= 3

fig, ax = plt.subplots(figsize=(14, 14), dpi=150)
ax.set_aspect("equal")
ax.axis("off")

ax.add_collection(lc)

ax.scatter(xy[mask_low, 0], xy[mask_low, 1],
           s=1.2, c="#c8c8c8", linewidths=0, zorder=2, alpha=0.4)

hub_sizes = np.sqrt(deg_arr[mask_hub]) * 5
ax.scatter(xy[mask_hub, 0], xy[mask_hub, 1],
           s=np.clip(hub_sizes, 5, 80),
           c=deg_arr[mask_hub], cmap="Blues",
           vmin=3, vmax=deg_arr.max(),
           linewidths=0.3, edgecolors="#555555",
           zorder=3)

# ── colorbar for edge span ─────────────────────────────────────────────────
sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, orientation="horizontal",
                  label="SNP distance (log scale)",
                  shrink=0.45, pad=0.02, aspect=30)
cb.ax.tick_params(labelsize=8)

# ── label top hubs with SNP index and degree ──────────────────────────────
TOP_N    = 8
top_hubs = sorted(degrees, key=degrees.get, reverse=True)[:TOP_N]
LABEL_BBOX = dict(boxstyle="round,pad=0.25", fc="white", ec="#aaaaaa", lw=0.6, alpha=0.9)
for node in top_hubs:
    x0, y0 = pos[node]
    r = np.hypot(x0, y0)
    label = f"SNP {node}\ndeg {degrees[node]}"
    if r < 1e-9:
        ax.annotate(label,
                    xy=(x0, y0), xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7, fontweight="bold", color="#111111",
                    ha="left", va="bottom", bbox=LABEL_BBOX)
    else:
        dx, dy = x0 / r, y0 / r
        push   = max(r * 0.10, 1.2)
        ax.text(x0 + dx * push, y0 + dy * push, label,
                fontsize=6.5, ha="center", va="center",
                color="#111111", fontweight="bold", bbox=LABEL_BBOX)

ax.autoscale()
fig.tight_layout()
fig.savefig("clt_tree.pdf", **PDF_KWARGS)
plt.close(fig)
print("Saved: clt_tree.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# 7b. COMPRESSED FULL TREE — same 10k nodes but with power-law radius compression
# ─────────────────────────────────────────────────────────────────────────────

pos_compressed = power_compress(pos, gamma=0.5)
# normalise to unit radius
max_r_comp = max(np.hypot(x, y) for x, y in pos_compressed.values()) or 1.0
pos_compressed = {n: np.array([x, y]) / max_r_comp for n, (x, y) in pos_compressed.items()}

xy_comp = np.array([pos_compressed[n] for n in range(K)])

segs_comp = [(pos_compressed[edge_list[i][0]], pos_compressed[edge_list[i][1]]) for i in edge_order]
colors_comp = edge_cmap(edge_norm(gaps_sorted))
lc_comp = LineCollection(segs_comp, linewidths=0.55, colors=colors_comp, alpha=0.75, zorder=1)

fig_comp, ax_comp = plt.subplots(figsize=(14, 14), dpi=150)
ax_comp.set_aspect("equal")
ax_comp.axis("off")
ax_comp.add_collection(lc_comp)

ax_comp.scatter(xy_comp[mask_low, 0], xy_comp[mask_low, 1],
                s=1.2, c="#c8c8c8", linewidths=0, zorder=2, alpha=0.4)
hub_sizes_comp = np.sqrt(deg_arr[mask_hub]) * 5
ax_comp.scatter(xy_comp[mask_hub, 0], xy_comp[mask_hub, 1],
                s=np.clip(hub_sizes_comp, 5, 80),
                c=deg_arr[mask_hub], cmap="Blues",
                vmin=3, vmax=deg_arr.max(),
                linewidths=0.3, edgecolors="#555555", zorder=3)

sm_comp = plt.cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
sm_comp.set_array([])
cb_comp = fig_comp.colorbar(sm_comp, ax=ax_comp, orientation="horizontal",
                            label="SNP distance (log scale)",
                            shrink=0.45, pad=0.02, aspect=30)
cb_comp.ax.tick_params(labelsize=8)

for node in top_hubs:
    x0, y0 = pos_compressed[node]
    r = np.hypot(x0, y0)
    label = f"SNP {node}\ndeg {degrees[node]}"
    if r < 1e-9:
        ax_comp.annotate(label, xy=(x0, y0), xytext=(3, 3),
                         textcoords="offset points",
                         fontsize=7, fontweight="bold", color="#111111",
                         ha="left", va="bottom", bbox=LABEL_BBOX)
    else:
        dx, dy = x0 / r, y0 / r
        push = max(r * 0.10, 0.02)
        ax_comp.text(x0 + dx * push, y0 + dy * push, label,
                     fontsize=6.5, ha="center", va="center",
                     color="#111111", fontweight="bold", bbox=LABEL_BBOX)

ax_comp.autoscale()
fig_comp.tight_layout()
fig_comp.savefig("clt_tree_compressed.pdf", **PDF_KWARGS)
plt.close(fig_comp)
print("Saved: clt_tree_compressed.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  COMPACT TREE — degree-2 chains collapsed to single edges
#
#     The 2962 chain nodes (degree exactly 2) are removed and their paths
#     replaced with a single edge between the two non-chain endpoints.
#     Only hub (deg ≥ 3) and leaf (deg = 1) nodes remain — 70% fewer nodes,
#     with the same branching structure and plasma coloring.
# ─────────────────────────────────────────────────────────────────────────────

def contract_chains(tree, deg_map):
    """Remove all degree-2 chain nodes, connecting their endpoints directly.
    Edge attribute snp_gap = |u_snp_index − v_snp_index| for the two endpoints."""
    keep = {n for n, d in deg_map.items() if d != 2}
    C = nx.Graph()
    C.add_nodes_from(keep)
    seen = set()
    for u in keep:
        for nb in tree.neighbors(u):
            ek = (min(u, nb), max(u, nb))
            if ek in seen:
                continue
            seen.add(ek)
            prev, curr = u, nb
            while curr not in keep:
                nxt = [x for x in tree.neighbors(curr) if x != prev]
                prev, curr = curr, nxt[0]
            if not C.has_edge(u, curr):
                C.add_edge(u, curr, snp_gap=abs(u - curr))
    return C

def align_and_normalise(fresh, reference_pos, root, tree):
    """Rotate `fresh` layout to match the orientation of `reference_pos`,
    using the highest-degree neighbour of root as the alignment anchor."""
    anchor = max(tree.neighbors(root), key=lambda n: degrees[n])
    theta_ref   = np.arctan2(reference_pos[anchor][1], reference_pos[anchor][0])
    theta_fresh = np.arctan2(fresh[anchor][1],         fresh[anchor][0])
    delta = theta_ref - theta_fresh
    c, s  = np.cos(delta), np.sin(delta)
    rotated = {n: np.array([x*c - y*s, x*s + y*c]) for n, (x, y) in fresh.items()}
    max_r = max(np.hypot(x, y) for x, y in rotated.values()) or 1.0
    return {n: np.array([x, y]) / max_r for n, (x, y) in rotated.items()}

C = contract_chains(T, degrees)
print(f"\nCompact tree: {C.number_of_nodes()} nodes, {C.number_of_edges()} edges "
      f"(removed {K - C.number_of_nodes()} degree-2 chain nodes)")

pos_c    = align_and_normalise(power_compress(radial_tree_layout(C, hub_root)), pos, hub_root, C)
nodes_c  = [n for n in range(K) if n in C]
xy_c     = np.array([pos_c[n] for n in nodes_c])
orig_deg = np.array([degrees[n] for n in nodes_c])

mask_low_c  = orig_deg <= 2   # leaf nodes (deg=1 in original)
mask_hub_c  = orig_deg >= 3   # hub nodes
hub_sizes_c = np.sqrt(orig_deg[mask_hub_c]) * 5

c_edges  = list(C.edges(data=True))
c_gaps   = np.array([d["snp_gap"] for u, v, d in c_edges])
c_order  = np.argsort(c_gaps)
c_segs   = [(pos_c[c_edges[i][0]], pos_c[c_edges[i][1]]) for i in c_order]
c_colors = edge_cmap(edge_norm(c_gaps[c_order]))

lc_c = LineCollection(c_segs, linewidths=0.55, colors=c_colors, alpha=0.75, zorder=1)

fig_c, ax_c = plt.subplots(figsize=(14, 14), dpi=150)
ax_c.set_aspect("equal")
ax_c.axis("off")
ax_c.add_collection(lc_c)

ax_c.scatter(xy_c[mask_low_c, 0], xy_c[mask_low_c, 1],
             s=1.2, c="#c8c8c8", linewidths=0, zorder=2, alpha=0.4)
ax_c.scatter(xy_c[mask_hub_c, 0], xy_c[mask_hub_c, 1],
             s=np.clip(hub_sizes_c, 5, 80),
             c=orig_deg[mask_hub_c], cmap="Blues",
             vmin=3, vmax=orig_deg.max(),
             linewidths=0.3, edgecolors="#555555", zorder=3)

sm_c = plt.cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
sm_c.set_array([])
cb_c = fig_c.colorbar(sm_c, ax=ax_c, orientation="horizontal",
                      label="SNP distance (log scale)",
                      shrink=0.45, pad=0.02, aspect=30)
cb_c.ax.tick_params(labelsize=8)

for node in top_hubs:
    if node not in pos_c:
        continue
    x0, y0 = pos_c[node]
    r = np.hypot(x0, y0)
    label = f"SNP {node}\ndeg {degrees[node]}"
    if r < 1e-9:
        ax_c.annotate(label, xy=(x0, y0), xytext=(3, 3),
                      textcoords="offset points",
                      fontsize=7, fontweight="bold", color="#111111",
                      ha="left", va="bottom", bbox=LABEL_BBOX)
    else:
        dx, dy = x0 / r, y0 / r
        push = max(r * 0.10, 0.02)
        ax_c.text(x0 + dx * push, y0 + dy * push, label,
                  fontsize=6.5, ha="center", va="center",
                  color="#111111", fontweight="bold", bbox=LABEL_BBOX)

ax_c.autoscale()
fig_c.tight_layout()
fig_c.savefig("clt_tree_compact.pdf", **PDF_KWARGS)
plt.close(fig_c)
print("Saved: clt_tree_compact.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  HUB NETWORK — single round of leaf pruning on the compact tree
#
#     Removes all degree-1 nodes from the compact tree C (the ~4599 original
#     leaf SNPs), then contracts any degree-2 chains that form.  Shows only
#     the hub-to-hub connectivity — nodes that have multiple connections to
#     other branching points.
# ─────────────────────────────────────────────────────────────────────────────

H = C.copy()
H.remove_nodes_from([n for n, d in H.degree() if d == 1])
H = contract_chains(H, dict(H.degree()))

print(f"\nHub network: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges "
      f"(removed {K - H.number_of_nodes()} nodes from original)")

hub_root_h = hub_root if hub_root in H else max(H.degree(), key=lambda x: x[1])[0]
pos_h   = align_and_normalise(power_compress(radial_tree_layout(H, hub_root_h)), pos, hub_root_h, H)
nodes_h = sorted(H.nodes())
xy_h    = np.array([pos_h[n] for n in nodes_h])
deg_h   = np.array([degrees[n] for n in nodes_h])
hub_sizes_h = np.sqrt(deg_h) * 5

h_edges  = list(H.edges(data=True))
h_gaps   = np.array([d["snp_gap"] for u, v, d in h_edges])
h_order  = np.argsort(h_gaps)
h_segs   = [(pos_h[h_edges[i][0]], pos_h[h_edges[i][1]]) for i in h_order]
h_colors = edge_cmap(edge_norm(h_gaps[h_order]))

lc_h = LineCollection(h_segs, linewidths=0.7, colors=h_colors, alpha=0.85, zorder=1)

fig_h, ax_h = plt.subplots(figsize=(14, 14), dpi=150)
ax_h.set_aspect("equal")
ax_h.axis("off")
ax_h.add_collection(lc_h)

ax_h.scatter(xy_h[:, 0], xy_h[:, 1],
             s=np.clip(hub_sizes_h, 5, 80),
             c=deg_h, cmap="Blues",
             vmin=3, vmax=deg_h.max(),
             linewidths=0.3, edgecolors="#555555", zorder=3)

sm_h = plt.cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
sm_h.set_array([])
cb_h = fig_h.colorbar(sm_h, ax=ax_h, orientation="horizontal",
                      label="SNP distance (log scale)",
                      shrink=0.45, pad=0.02, aspect=30)
cb_h.ax.tick_params(labelsize=8)

for node in top_hubs:
    if node not in pos_h:
        continue
    x0, y0 = pos_h[node]
    r = np.hypot(x0, y0)
    label = f"SNP {node}\ndeg {degrees[node]}"
    if r < 1e-9:
        ax_h.annotate(label, xy=(x0, y0), xytext=(3, 3),
                      textcoords="offset points",
                      fontsize=7, fontweight="bold", color="#111111",
                      ha="left", va="bottom", bbox=LABEL_BBOX)
    else:
        dx, dy = x0 / r, y0 / r
        push = max(r * 0.10, 0.02)
        ax_h.text(x0 + dx * push, y0 + dy * push, label,
                  fontsize=6.5, ha="center", va="center",
                  color="#111111", fontweight="bold", bbox=LABEL_BBOX)

ax_h.autoscale()
fig_h.tight_layout()
fig_h.savefig("clt_tree_hubs.pdf", **PDF_KWARGS)
plt.close(fig_h)
print("Saved: clt_tree_hubs.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# 10. SLIDES VARIANT — degree distribution with consistent palette + big text
#     Colors match allcolpal: GPC/CLT = "blue", HMM = "purple"
# ─────────────────────────────────────────────────────────────────────────────

CLT_COLOR_SL = "blue"    # matches GPC in allcolpal
HMM_COLOR_SL = "purple"  # matches HMM in allcolpal

FONTSIZE_SL_LABEL  = 18
FONTSIZE_SL_TICK   = 15
FONTSIZE_SL_LEGEND = 15

fig_sl, ax_sl = plt.subplots(figsize=(7, 5))
ax_sl.bar(x_pos - width/2, hmm_counts, width, color=HMM_COLOR_SL,
          label="HMM (chain)", edgecolor="white", linewidth=0.5, alpha=0.85)
ax_sl.bar(x_pos + width/2, clt_counts, width, color=CLT_COLOR_SL,
          label="GPC (Chow-Liu tree)", edgecolor="white", linewidth=0.5)

ax_sl.set_xticks(x_pos)
ax_sl.set_xticklabels(tick_labels, fontsize=FONTSIZE_SL_TICK)
ax_sl.tick_params(axis="y", labelsize=FONTSIZE_SL_TICK)
ax_sl.set_xlabel("Degree", fontsize=FONTSIZE_SL_LABEL)
ax_sl.set_ylabel("Number of SNPs", fontsize=FONTSIZE_SL_LABEL)
ax_sl.legend(frameon=False, fontsize=FONTSIZE_SL_LEGEND)

fig_sl.tight_layout()
fig_sl.savefig("clt_degree_distribution_slides.pdf", **PDF_KWARGS)
plt.close(fig_sl)
print("\nSaved: clt_degree_distribution_slides.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# 11. SLIDES VARIANT — compact tree with bigger colorbar + bigger SNP labels
# ─────────────────────────────────────────────────────────────────────────────

LABEL_BBOX_SL = dict(boxstyle="round,pad=0.55", fc="white", ec="#888888",
                     lw=1.2, alpha=0.95)
LABEL_FONTSIZE_SL = 14

lc_c_sl = LineCollection(c_segs, linewidths=0.55, colors=c_colors, alpha=0.75, zorder=1)

fig_csl, ax_csl = plt.subplots(figsize=(14, 14), dpi=150)
ax_csl.set_aspect("equal")
ax_csl.axis("off")
ax_csl.add_collection(lc_c_sl)

ax_csl.scatter(xy_c[mask_low_c, 0], xy_c[mask_low_c, 1],
               s=1.2, c="#c8c8c8", linewidths=0, zorder=2, alpha=0.4)
ax_csl.scatter(xy_c[mask_hub_c, 0], xy_c[mask_hub_c, 1],
               s=np.clip(hub_sizes_c, 5, 80),
               c=orig_deg[mask_hub_c], cmap="Blues",
               vmin=3, vmax=orig_deg.max(),
               linewidths=0.3, edgecolors="#555555", zorder=3)

sm_csl = plt.cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
sm_csl.set_array([])
cb_csl = fig_csl.colorbar(sm_csl, ax=ax_csl, orientation="horizontal",
                           shrink=0.65, pad=0.03, aspect=18)
cb_csl.set_label("SNP distance (log scale)", fontsize=22)
cb_csl.ax.tick_params(labelsize=20)
cb_csl.ax.xaxis.label.set_size(22)

for node in top_hubs:
    if node not in pos_c:
        continue
    x0, y0 = pos_c[node]
    r = np.hypot(x0, y0)
    label = f"SNP {node}\ndeg {degrees[node]}"
    if r < 1e-9:
        ax_csl.annotate(label, xy=(x0, y0), xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=LABEL_FONTSIZE_SL, fontweight="bold",
                        color="#111111", ha="left", va="bottom",
                        bbox=LABEL_BBOX_SL)
    else:
        dx, dy = x0 / r, y0 / r
        push = max(r * 0.12, 0.05)
        ax_csl.text(x0 + dx * push, y0 + dy * push, label,
                    fontsize=LABEL_FONTSIZE_SL, ha="center", va="center",
                    color="#111111", fontweight="bold", bbox=LABEL_BBOX_SL)

ax_csl.autoscale()
fig_csl.tight_layout()
fig_csl.savefig("clt_tree_compact_slides.pdf", **PDF_KWARGS)
plt.close(fig_csl)
print("Saved: clt_tree_compact_slides.pdf")


print("\nDone.")