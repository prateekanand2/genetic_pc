# GPC: An expressive and tractable deep generative model for genetic variation data

<p align="center">
  <img src="assets/clt_tree.png" alt="Chow-Liu tree backbone of a GPC trained on 1000 Genomes SNPs" width="400">
</p>

**GPC** (Genetic Probabilistic Circuit) is a tractable deep generative model for haplotype data. It supports exact likelihood evaluation, exact marginalization, and fast conditional queries. The same trained model can generate artificial genomes and impute missing SNPs with improved privacy properties.

Preprint: [GPC: An expressive and tractable deep generative model for genetic variation data (bioRxiv, 2026)](https://www.biorxiv.org/content/10.1101/2023.05.16.541036v3).

---

## Installation

```bash
git clone https://github.com/sriramlab/GPC.git
cd GPC

pip install pyjuice
```

Requires a CUDA-capable GPU, PyTorch, NumPy, pandas, scikit-learn, networkx, matplotlib, and tqdm. GPC is built on [PyJuice](https://github.com/Tractables/pyjuice).

### A patch is needed when many SNPs are invariant

PyJuice's product-layer kernel reads out of bounds when a product node has more than 2,048 children. The symptom is a CUDA *illegal memory access* during the first training step.

This is reached through the data, not the region size. A monomorphic column has zero entropy, so its mutual information with every other SNP is zero; the Chow–Liu step gains nothing by placing it anywhere in particular and attaches such columns to a single hub, giving that node one child per invariant SNP. More than 2,048 invariant SNPs in the training matrix is therefore enough to trip it. That is easy to reach in an ancestrally narrow or small subset — the 1KG African training set here has 3,722 invariant sites out of 10,000 — while the cosmopolitan sets have far fewer (3.1% for 1KG 8020, none for the high-coverage region) and never trip it. Two one-line changes fix it, and `benchmarks/prod_layer_2048_fix.patch` applies both:

```bash
git clone https://github.com/Tractables/pyjuice.git
cd pyjuice
git apply /path/to/GPC/benchmarks/prod_layer_2048_fix.patch
pip install -e .
```

```python
# src/pyjuice/layer/prod_layer.py -- the loop trip count for the >2048 path
-                N_NUM_BLKS = triton.cdiv(num_edges, BLOCK_B),
+                N_NUM_BLKS = triton.cdiv(num_edges, BLOCK_N),

# src/pyjuice/layer/kernels/prod.py -- inside _forward_backward_kernel_large
-        offs_egstart = tl.load(cids_ptr + nblock_id * num_edges + offs_edge)
+        offs_egstart = tl.load(cids_ptr + nblock_id * num_edges + offs_edge,
+                               mask = mask_edge, other = 0)
```

This path sets `BLOCK_N = 2048` and `BLOCK_B = 1`, so dividing by `BLOCK_B` makes the loop run once per edge instead of once per 2,048-edge tile; the second change adds the bounds mask the surrounding code already computes.

Filtering invariant columns before training also avoids the condition, and is worth doing regardless since such columns carry no information. `benchmarks/prod_layer_large_bug.py` exercises the kernel directly at a chosen fan-in.

---

## Data format

A whitespace-separated `0/1` haplotype file with no header: rows = haplotypes, columns = SNPs. Genotypes must be phased and split into haplotypes, so a diploid cohort contributes two rows per individual. Columns should be a contiguous genomic region in position order — the Chow–Liu backbone learns from linkage disequilibrium between neighbouring SNPs, so a randomly ordered or randomly sampled set of SNPs will train but produce a poor model.

Optionally, a `.legend` file with header `id position a0 a1` (space-separated) giving the bp position of each SNP. If supplied, LD plots use bp distance on the x-axis.

`pc/demo/` ships two files from a contiguous chr15 region in 1000 Genomes Project Phase 3 (5008 haplotypes):

| File | SNPs | Use for |
|---|---|---|
| `1K_full.txt` + `1K_full.legend`   | 1,000  | default — fast end-to-end run |
| `10K_full.txt` + `10K_full.legend` | 10,000 | full run |

---

## Quick start (1K SNPs, ~ a few minutes on one GPU)

From `pc/demo/`, run the four steps in order. Every script shares one `--run-dir` (default `out/1K`).

```bash
cd pc/demo

# 1. Train a GPC with train/val/test split and early stopping.
python3 train_demo.py

# 2. Sample artificial genomes from the best checkpoint.
python3 generate_demo.py

# 3. Evaluate sample quality + privacy: PCA, LD decay, LD error, CLT tree, AATS.
python3 evaluate.py

# 4. Imputation benchmark: single-SNP + multi-SNP at 30/50/80% missingness.
python3 impute_demo.py
python3 plot_imputation.py
```

Afterwards `out/1K/` contains:

```
out/1K/
├── config.json
├── gpc_best.jpc                   best-val GPC checkpoint
├── train.txt / val.txt / test.txt shuffled splits (reused downstream)
├── train.log                      per-epoch train/val LL
├── samples.txt                    generated haplotypes
├── quality/                       pca, ld_decay, ld_error, clt_tree, clt_summary
├── imputation/                    r2 CSVs + imputation_r2.pdf + imputation_summary.csv
└── privacy/                       aats
```

All scripts take `--help`.

## Full run (10K SNPs)

Same four steps, bigger model:

```bash
cd pc/demo

python3 train_demo.py      --data 10K_full.txt --output-dir out/10K \
    --latents 128 --epochs 2000 --patience 100 --seed 1
python3 generate_demo.py   --run-dir out/10K --num-samples 5008 --seed 1
python3 evaluate.py        --run-dir out/10K --legend 10K_full.legend --seed 1
python3 impute_demo.py     --run-dir out/10K --mask-rates 0.3 0.5 0.8 --seed 1
python3 plot_imputation.py --run-dir out/10K
```

## Using your own data

```bash
python3 train_demo.py --data path/to/haps.txt --output-dir out/my_run \
    --latents 128 --epochs 2000 --patience 100 --seed 1
python3 generate_demo.py   --run-dir out/my_run
python3 evaluate.py        --run-dir out/my_run --legend path/to/haps.legend
python3 impute_demo.py     --run-dir out/my_run
python3 plot_imputation.py --run-dir out/my_run
```

Pass `--legend ''` to `evaluate.py` to fall back to SNP-index distance when no legend is available.

### Choosing settings

| Setting | Guidance |
|---|---|
| `--latents` | 128 throughout the paper. Larger values did not improve held-out likelihood and cost more memory; 16 is enough for the 1K demo. |
| `--epochs` / `--patience` | Training stops on held-out log-likelihood, so set these generously and let early stopping decide. This is the main practical advantage over GAN- and RBM-based models, which have no such signal. |
| Region size | Up to roughly 10,000–15,000 SNPs on a standard workstation. The binding constraint is *host* RAM while the Chow–Liu tree is built, which grows with the square of the number of SNPs — not GPU memory. |
| GPU | One card is enough. The largest model in the paper peaks at about 5 GB of GPU memory. |

To model a longer region, split it into contiguous chunks and train one model per chunk.

Measured training cost on one NVIDIA RTX A5000, at 4,006 haplotypes and 12,287 SNPs:

| | Total training | Peak GPU memory |
|---|---|---|
| GPC | 3.3 h | 4.7 GB |
| WGAN | 22.8 h | 12.8 GB |
| RBM | 35.3 h | 1.3 GB |

GPC is not the cheapest per epoch — an RBM epoch is faster — but it converges in far fewer epochs. `benchmarks/` contains the script that produced this table.

---

## Repository layout

```
pc/demo/          self-contained demo (start here)
pc/               training / sampling / imputation scripts used in the paper
benchmarks/       runtime and memory scaling benchmark for GPC, RBM and WGAN
plots/            analysis notebooks and figure-generating scripts for the paper
aux/              SNP legends, MAF tables, AATS utilities
impute5/          wrappers for the Impute5 reference-panel comparisons
results/          per-dataset outputs (metrics, VCFs, logs)
```

Model checkpoints (`.jpc`) and raw sample matrices are not tracked: they are large and
regenerated by the scripts above.

`plots/structure/` adopts and extends code from Yelmen et al., *Deep convolutional and conditional neural networks for large-scale genomic data generation* ([PLOS Comput. Biol.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011584)).

---

## Citation

```
@article{anand2026gpc,
  title   = {GPC: An expressive and tractable deep generative model for genetic variation data},
  author  = {Anand, Prateek and Liu, Anji and Dang, Meihua and Fu, Boyang and Wei, Xinzhu and Van den Broeck, Guy and Sankararaman, Sriram},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.1101/2023.05.16.541036},
  url     = {https://www.biorxiv.org/content/10.1101/2023.05.16.541036v3}
}
```
