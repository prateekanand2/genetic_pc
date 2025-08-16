import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse

def extract_test_genotype_array(vcf_file, snp_set):
    results = {}
    with open(vcf_file, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            snp_id = f"{fields[0]}:{fields[1]}"
            if snp_id in snp_set:
                genotypes = []
                for g in fields[9:]:
                    if g == "0":
                        genotypes.append(0)
                    elif g == "1":
                        genotypes.append(1)
                    else:
                        genotypes.append(-1)
                results[snp_id] = genotypes
    return results

def compute_r2(dosages, genotypes, snp):
    if len(dosages) != len(genotypes):
        raise ValueError(f"Length mismatch for {snp}: dosages={len(dosages)}, genotypes={len(genotypes)}")
    r, _ = pearsonr(dosages, genotypes)
    return r ** 2

def process_snp_file(filename, dosage_dir, vcf_file, bootstrap_vcfs, bootstrap_indices):
    snp = filename.replace(".txt", "")
    filepath = os.path.join(dosage_dir, filename)
    r2_boots = []

    try:
        dosages = np.loadtxt(filepath)
    except Exception as e:
        print(f"❌ Could not load dosages for {snp}: {e}")
        return (snp, None, *([None] * len(bootstrap_vcfs)))

    try:
        gt_main_dict = extract_test_genotype_array(vcf_file, [snp])
        if snp not in gt_main_dict:
            print(f"⚠️ SNP {snp} not found in main VCF: {vcf_file}")
            return (snp, None, *([None] * len(bootstrap_vcfs)))
        gt_main = np.array(gt_main_dict[snp])
        r2_main = compute_r2(dosages, gt_main, snp)
    except Exception as e:
        print(f"❌ Error computing R2 for main VCF for {snp}: {e}")
        return (snp, None, *([None] * len(bootstrap_vcfs)))

    for b_idx, (b_vcf, indices) in enumerate(zip(bootstrap_vcfs, bootstrap_indices)):
        try:
            if indices is None:
                r2_boots.append(None)
                continue

            gt_boot_dict = extract_test_genotype_array(b_vcf, [snp])
            if snp not in gt_boot_dict:
                print(f"⚠️ SNP {snp} not found in bootstrap VCF: {b_vcf}")
                r2_boots.append(None)
                continue
            gt_boot = np.array(gt_boot_dict[snp])
            r2_boot = compute_r2(dosages[indices], gt_boot, snp)
            r2_boots.append(r2_boot)
        except Exception as e:
            print(f"❌ Error computing R2 for SNP {snp} in {b_vcf}: {e}")
            r2_boots.append(None)

    return (snp, r2_main, *r2_boots)

def main(dosage_dir, vcf_file, bootstrap_vcf_dir, output_csv, n_bootstraps=10, max_workers=8):
    bootstrap_vcfs = [os.path.join(bootstrap_vcf_dir, f"bootstrap_{i}.vcf") for i in range(1, n_bootstraps + 1)]
    bootstrap_indices = []

    for i in range(1, n_bootstraps + 1):
        index_file = os.path.join(bootstrap_vcf_dir, f"indices_{i}.txt")
        try:
            indices = np.loadtxt(index_file, dtype=int)
        except Exception as e:
            print(f"❌ Failed to load indices from {index_file}: {e}")
            indices = None
        bootstrap_indices.append(indices)

    dosage_files = [f for f in os.listdir(dosage_dir) if f.endswith(".txt")]
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_snp_file, f, dosage_dir, vcf_file, bootstrap_vcfs, bootstrap_indices): f
            for f in dosage_files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing SNPs"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"❌ Unexpected error in processing: {e}")

    columns = ["SNP Set", "R2"] + [f"R2_boot_{i}" for i in range(1, n_bootstraps + 1)]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ R² results saved to: {output_csv}")

if __name__ == "__main__":
    # 🔧 === Edit these paths as needed ===
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    args = parser.parse_args()

    method = args.method

    dosage_dir = f"results/bootstrap/{method}_dosages"
    vcf_file = "../results/1KG/8020/data/8020_test.vcf.gz"
    bootstrap_vcf_dir = "../results/1KG/8020/data/test_bootstraps"
    output_csv = f"results/bootstrap/{method}_bootstraps.csv"
    n_bootstraps = 10
    max_workers = 16  # adjust based on available cores
    # ===================================

    main(dosage_dir, vcf_file, bootstrap_vcf_dir, output_csv, n_bootstraps=n_bootstraps, max_workers=max_workers)