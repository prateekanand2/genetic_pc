import pyjuice as juice
import torch
# import torchvision
import time
from torch.utils.data import TensorDataset, DataLoader
import pyjuice.nodes.distributions as dists
import numpy as np
import pandas as pd

import seaborn as sns
import pandas as pd
import numpy as np
import importlib
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from bed_reader import open_bed
import gc
from joblib import Parallel, delayed

def compute_r_squared_old(array1, array2):
    # Ensure both arrays are NumPy arrays
    array1 = np.array(array1)
    array2 = np.array(array2)

    # Calculate the correlation coefficient
    correlation_matrix = np.corrcoef(array1, array2)
    correlation_xy = correlation_matrix[0, 1]
    
    # Calculate R^2
    r_squared = correlation_xy ** 2
    
    return r_squared

def vcf_to_haplotype_array(vcf_file):
    haplotypes = []

    with open(vcf_file, 'r') as file:
        for line in file:
            # Skip header lines starting with '##' or the CHROM/POS line
            if line.startswith('##'):
                continue
            elif line.startswith('#CHROM'):
                # The first line starting with '#CHROM' contains sample names
                header = line.strip().split('\t')
                continue

            # Extract genotype data for each SNP
            fields = line.strip().split('\t')
            genotypes = fields[9:]  # Genotype data starts from the 10th column
            
            # Convert genotypes to integers (0 or 1)
            genotype_row = [int(genotype) for genotype in genotypes]
            
            # Append the row for this SNP
            haplotypes.append(genotype_row)

    # Convert the list of haplotypes into a NumPy array
    haplotype_array = np.array(haplotypes)
    
    # Return as-is (rows = SNPs, columns = samples)
    return haplotype_array.T

print("Number of CUDA devices:", torch.cuda.device_count())
print(torch.version.cuda)

device = torch.device("cuda:0")
np.random.seed(1)

print(device)

def compute_r_squared_per_feature(true_values, predicted_values):
    true_values = true_values.cpu().numpy()
    predicted_values = predicted_values.cpu().numpy()
    num_features = true_values.shape[1]
    r2_scores = np.zeros(num_features)
    for i in range(num_features):
        correlation_matrix = np.corrcoef(true_values[:, i], predicted_values[:, i])
        correlation_xy = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0
        r2_scores[i] = correlation_xy ** 2
    return r2_scores

def run_r2_on_data(valid_data_tensor, pc_model):
    dataset = TensorDataset(valid_data_tensor)
    dataloader = DataLoader(dataset, batch_size=512)
    num_samples = valid_data_tensor.size(0)
    num_features = valid_data_tensor.size(1)

    all_expected_counts = torch.zeros(num_samples, num_features, device="cpu")
    # all_predictions = torch.zeros(num_samples, num_features, device="cpu")
    all_originals = torch.zeros(num_samples, num_features, device="cpu")

    num_masked = 1
    batch_start = 0

    for batch in dataloader:
        data = batch[0].to(device)
        batch_size = data.size(0)
        batch_end = batch_start + batch_size

        for pos in range(0, num_features, num_masked):
            end_pos = min(pos + num_masked, num_features)

            false_array = torch.full((num_features,), False, dtype=torch.bool).to(device)
            false_array[pos:end_pos] = True
            missing_mask = false_array

            with torch.cuda.device("cuda:0"):
                lls = juice.queries.conditional(pc_model, data=data, missing_mask=missing_mask)
            probs = lls[:, pos:end_pos, :]

            for idx, feature_pos in enumerate(range(pos, end_pos)):
                original = data[:, feature_pos]
                prob_0 = probs[:, idx, 0]
                prob_1 = probs[:, idx, 1]
                expected_counts = prob_0 * 0 + prob_1 * 1

                # _, predictions = torch.max(probs[:, idx:idx+1, :], dim=-1)
                # predictions = predictions.squeeze()

                all_expected_counts[batch_start:batch_end, feature_pos] = expected_counts.detach().cpu()
                # all_predictions[batch_start:batch_end, feature_pos] = predictions.detach().cpu()
                all_originals[batch_start:batch_end, feature_pos] = original.detach().cpu()

        batch_start = batch_end

    r2s = compute_r_squared_per_feature(all_originals, all_expected_counts)
    return r2s

# === Load and compile PC model ===
pc_path = '../results/1KG/8020/hclt/pc_10K_8020_nooverlap_4006-128_5000epochs_ps0.005.jpc'
pc_model = juice.compile(juice.load(pc_path))
pc_model.to(device)

# === Step 1: Compute base R2 ===
print("Computing base R2...")
base_data_path = '../results/1KG/8020/data/8020_test.txt'
valid_data = np.loadtxt(base_data_path, dtype=np.int8, delimiter=' ')
valid_data_tensor = torch.tensor(valid_data, dtype=torch.long)
r2_base = run_r2_on_data(valid_data_tensor, pc_model)

# === Step 2: Compute bootstrap R2s ===
r2_boots = []
for boot_id in range(1, 11):
    print(f"Processing bootstrap_{boot_id}.vcf")
    file_path = f"../results/1KG/8020/data/test_bootstraps/bootstrap_{boot_id}.vcf"
    valid_data = vcf_to_haplotype_array(file_path)
    valid_data_tensor = torch.tensor(valid_data, dtype=torch.long)
    r2_boot = run_r2_on_data(valid_data_tensor, pc_model)
    r2_boots.append(r2_boot)

r2_boot_array = np.stack(r2_boots, axis=-1)  # shape (num_features, 10)

# === Step 3: Read legend and build DataFrame ===
legend_path = "../aux/10K_legend.maf.txt"
legend_df = pd.read_csv(legend_path, sep='\t', header=None, names=["SNP Set", "MAF"])

# === Step 4: Build final DataFrame ===
output_df = legend_df.copy()
output_df["R2"] = r2_base

for boot_id in range(1, 11):
    output_df[f"R2_boot_{boot_id}"] = r2_boot_array[:, boot_id - 1]

# Reorder columns: SNP Set, R2, R2_boot_1...R2_boot_10, MAF
cols = ["SNP Set", "R2"] + [f"R2_boot_{i}" for i in range(1, 11)] + ["MAF"]
output_df = output_df[cols]

# === Step 5: Save final CSV ===
output_path = "results/r2/bootstrap/10K_pc_real_8020_bootstraps.csv"
output_df.to_csv(output_path, index=False)
print(f"Saved output to {output_path}")