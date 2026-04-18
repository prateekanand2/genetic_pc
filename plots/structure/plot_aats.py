import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

data = {
    "1KG": {
        "AA_TRUTH (Train)": {"RBM": 0.9561, "WGAN": 0.8103, "GPC": 0.7185},
        "AA_TRUTH (Test)":  {"RBM": 0.9916, "WGAN": 0.8183, "GPC": 0.8199},
        "AA_SYN (Train)":   {"RBM": 0.0024, "WGAN": 0.7356, "GPC": 0.4225},
        "AA_SYN (Test)":    {"RBM": 0.0212, "WGAN": 0.7680, "GPC": 0.4912},
    },
    "UKBB": {
        "AA_TRUTH (Train)": {"RBM": 0.9934, "WGAN": 0.9664, "GPC": 0.9096},
        "AA_TRUTH (Test)":  {"RBM": 0.9970, "WGAN": 0.9692, "GPC": 0.9240},
        "AA_SYN (Train)":   {"RBM": 0.0094, "WGAN": 0.7828, "GPC": 0.5380},
        "AA_SYN (Test)":    {"RBM": 0.0114, "WGAN": 0.7730, "GPC": 0.4686},
    },
}

methods = ["RBM", "WGAN", "GPC"]
colors = {
    "RBM":  "orange",
    "WGAN": "green",
    "GPC":  "blue",
}

metrics = ["AA_TRUTH (Train)", "AA_TRUTH (Test)", "AA_SYN (Train)", "AA_SYN (Test)"]
metric_labels = [
    r"$AA_{\mathrm{TRUTH}}$" + "\n(Train)",
    r"$AA_{\mathrm{TRUTH}}$" + "\n(Test)",
    r"$AA_{\mathrm{SYN}}$"   + "\n(Train)",
    r"$AA_{\mathrm{SYN}}$"   + "\n(Test)",
]

n_metrics = len(metrics)
n_methods = len(methods)
bar_width  = 0.22
group_gap  = 0.12
x_positions = []
pos = 0.0
for i in range(n_metrics):
    x_positions.append(pos)
    pos += n_methods * bar_width + group_gap

# --- combined plot ---
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
fig.subplots_adjust(wspace=0.06)

for ax, dataset in zip(axes, ["1KG", "UKBB"]):
    for mi, metric in enumerate(metrics):
        xc = x_positions[mi]
        for ji, method in enumerate(methods):
            val = data[dataset][metric][method]
            dev = abs(val - 0.5)
            xbar = xc + ji * bar_width
            ax.bar(xbar, dev, width=bar_width,
                   color=colors[method], zorder=3, linewidth=0)
            ax.text(xbar, dev + 0.008,
                    f"{dev:.3f}", ha="center", va="bottom",
                    fontsize=10, color="#333333")

    tick_positions = [xp + bar_width for xp in x_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(metric_labels, fontsize=15)
    ax.set_ylim(0, 0.58)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.tick_params(axis="y", labelsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.15, x_positions[-1] + n_methods * bar_width + 0.30)
    ax.set_xlabel(dataset, fontsize=16, fontweight="bold", labelpad=8)

axes[0].set_ylabel(r"Absolute Error ($|AA - 0.5|$)", fontsize=16)

legend_handles = [mpatches.Patch(color=colors[m], label=m) for m in methods]
fig.legend(handles=legend_handles, loc="upper center",
           ncol=3, fontsize=14, frameon=False,
           bbox_to_anchor=(0.5, 1.03),
           handlelength=1.8, handleheight=1.0)

plt.savefig("aats.pdf", bbox_inches="tight", dpi=300)
plt.close()
print("Saved aats.pdf")

# --- separate plots ---
for dataset in ["1KG", "UKBB"]:
    fig, ax = plt.subplots(figsize=(8, 5))

    for mi, metric in enumerate(metrics):
        xc = x_positions[mi]
        for ji, method in enumerate(methods):
            val = data[dataset][metric][method]
            dev = abs(val - 0.5)
            xbar = xc + ji * bar_width
            ax.bar(xbar, dev, width=bar_width,
                   color=colors[method], zorder=3, linewidth=0)
            ax.text(xbar, dev + 0.008,
                    f"{dev:.3f}", ha="center", va="bottom",
                    fontsize=10, color="#333333")

    tick_positions = [xp + bar_width for xp in x_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(metric_labels, fontsize=15)

    ax.set_ylim(0, 0.58)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel(r"Absolute Error ($|AA - 0.5|$)", fontsize=16)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.15, x_positions[-1] + n_methods * bar_width + 0.30)

    legend_handles = [mpatches.Patch(color=colors[m], label=m) for m in methods]
    ax.legend(handles=legend_handles, loc="upper right",
              ncol=1, fontsize=14, frameon=False,
              handlelength=1.6, handleheight=1.0)

    fname = f"aats_{dataset.lower()}.pdf"
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved {fname}")
