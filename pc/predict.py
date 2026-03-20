import pyjuice as juice
import torch
import time
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os


def vcf_to_haplotype_array(vcf_file):
    haplotypes = []
    with open(vcf_file, 'r') as file:
        for line in file:
            if line.startswith('##'):
                continue
            elif line.startswith('#CHROM'):
                continue
            fields = line.strip().split('\t')
            genotypes = fields[9:]
            genotype_row = [int(genotype) for genotype in genotypes]
            haplotypes.append(genotype_row)
    return np.array(haplotypes).T


def compute_r_squared_per_feature(true_values, predicted_values):
    """Vectorized R^2 across all features at once."""
    t_mean = true_values.mean(axis=0)
    p_mean = predicted_values.mean(axis=0)
    t_c = true_values - t_mean
    p_c = predicted_values - p_mean
    numer = (t_c * p_c).sum(axis=0)
    denom = np.sqrt((t_c ** 2).sum(axis=0) * (p_c ** 2).sum(axis=0))
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.where(denom > 0, numer / denom, 0.0)
    return corr ** 2


def run_r2_on_data(valid_data_tensor, pc_model, batch_size, desc=""):
    dataset = TensorDataset(valid_data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    num_samples = valid_data_tensor.size(0)
    num_features = valid_data_tensor.size(1)

    all_prob1 = torch.zeros(num_samples, num_features)
    missing_mask = torch.zeros(num_features, dtype=torch.bool, device=device)

    batch_start = 0
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx, batch in enumerate(dataloader):
        data = batch[0].to(device)
        bs = data.size(0)
        batch_end = batch_start + bs

        pbar = tqdm(range(num_features),
                     desc=f"{desc} batch {batch_idx+1}/{num_batches}",
                     unit="snp", leave=False)
        for pos in pbar:
            missing_mask[pos] = True

            with torch.cuda.device("cuda"):
                lls = juice.queries.conditional(pc_model, data=data, missing_mask=missing_mask)

            all_prob1[batch_start:batch_end, pos] = lls[:, pos, 1].detach().cpu()
            missing_mask[pos] = False

        batch_start = batch_end

    originals = valid_data_tensor.float().numpy()
    predictions = all_prob1.numpy()
    return compute_r_squared_per_feature(originals, predictions)


def save_progress(output_df, cols, output_path):
    available_cols = [c for c in cols if c in output_df.columns]
    output_df[available_cols].to_csv(output_path, index=False)


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────
# Usage: python3 predict_bootstrap.py <chrom> [batch_size]
#   batch_size defaults to 512 if not provided

print("Number of CUDA devices:", torch.cuda.device_count())
print(torch.version.cuda)

device = torch.device("cuda")
np.random.seed(1)

chrom = sys.argv[1]
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 512
print(f"Using batch_size={batch_size}")

# === Load and compile PC model ===
pc_path = f'/scratch2/prateek/genetic_pc_github/results/UKBB/8020/hclt/pc_ukbb_8020_chr{chrom}_21538-128_2000epochs_ps0.005.jpc'
print(f"Loading model from {pc_path} ...")
pc_model = juice.compile(juice.load(pc_path))
pc_model.to(device)
print("Model loaded and compiled.")

# === Prepare output ===
legend_path = f"../aux/chr{chrom}_legend.maf.txt"
legend_df = pd.read_csv(legend_path, sep='\t', header=None, names=["SNP Set", "MAF"])
output_path = f"/scratch2/prateek/genetic_pc_github/plots/impute/results/r2/UKBB_pc_chr{chrom}_8020_bootstraps.csv"
cols = ["SNP Set", "R2"] + [f"R2_boot_{i}" for i in range(1, 11)] + ["MAF"]

# === Check for existing progress ===
completed = set()
if os.path.exists(output_path):
    existing_df = pd.read_csv(output_path)
    output_df = legend_df.copy()
    # Restore any columns that are already complete
    if "R2" in existing_df.columns and existing_df["R2"].notna().all():
        output_df["R2"] = existing_df["R2"].values
        completed.add("R2")
    for i in range(1, 11):
        col = f"R2_boot_{i}"
        if col in existing_df.columns and existing_df[col].notna().all():
            output_df[col] = existing_df[col].values
            completed.add(col)
    if completed:
        print(f"Resuming — already completed: {sorted(completed)}")
else:
    output_df = legend_df.copy()

# === Step 1: Base R2 ===
if "R2" not in completed:
    print("Computing base R2...")
    base_data_path = f'/scratch2/prateek/genetic_pc_github/results/UKBB/8020/data/other_chrs/{chrom}/chr{chrom}_test.txt'
    valid_data = np.loadtxt(base_data_path, dtype=np.int8, delimiter=' ')
    valid_data_tensor = torch.tensor(valid_data, dtype=torch.long)

    t0 = time.perf_counter()
    r2_base = run_r2_on_data(valid_data_tensor, pc_model, batch_size, desc="Base")
    print(f"Base R2 done in {(time.perf_counter() - t0) / 60:.1f} minutes.")

    output_df["R2"] = r2_base
    save_progress(output_df, cols, output_path)
    print(f"  -> Saved progress to {output_path}")
else:
    print("Skipping base R2 (already complete)")

# === Step 2: Bootstrap R2s ===
for boot_id in range(1, 11):
    col = f"R2_boot_{boot_id}"
    if col in completed:
        print(f"Skipping bootstrap {boot_id} (already complete)")
        continue

    print(f"Processing bootstrap_{boot_id}.vcf")
    file_path = f"/scratch2/prateek/genetic_pc_github/results/UKBB/8020/data/other_chrs/{chrom}/test_bootstraps/bootstrap_{boot_id}.vcf"
    valid_data = vcf_to_haplotype_array(file_path)
    valid_data_tensor = torch.tensor(valid_data, dtype=torch.long)

    t0 = time.perf_counter()
    r2_boot = run_r2_on_data(valid_data_tensor, pc_model, batch_size, desc=f"Boot {boot_id}")
    print(f"Bootstrap {boot_id} done in {(time.perf_counter() - t0) / 60:.1f} minutes.")

    output_df[col] = r2_boot
    save_progress(output_df, cols, output_path)
    print(f"  -> Saved progress to {output_path}")

print(f"All done. Final output at {output_path}")