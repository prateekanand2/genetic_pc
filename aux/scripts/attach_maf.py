import pandas as pd
import os

# paths
result_dir = "/scratch2/prateek/genetic_pc_github/plots/impute/results/r2"
maf_dir = "/scratch2/prateek/genetic_pc_github/aux"

chroms = [2, 11, 12, 14, 18]
methods = ["real", "pc_real", "hclt", "rbm"]

for chrnum in chroms:

    maf_file = f"{maf_dir}/chr{chrnum}_legend.maf.txt"

    if not os.path.exists(maf_file):
        print(f"Missing MAF file: {maf_file}")
        continue

    maf_df = pd.read_csv(
        maf_file,
        sep="\t",
        header=None,
        names=["SNP Set", "MAF"]
    )

    maf_map = dict(zip(maf_df["SNP Set"], maf_df["MAF"]))

    for method in methods:

        result_file = f"{result_dir}/UKBB_{method}_chr{chrnum}_8020_bootstraps.csv"

        if not os.path.exists(result_file):
            print(f"Skipping missing file: {result_file}")
            continue

        print(f"Processing {result_file}")

        df = pd.read_csv(result_file)

        # attach MAF
        df["MAF"] = df["SNP Set"].map(maf_map)

        missing = df["MAF"].isna().sum()
        if missing > 0:
            print(f"⚠ {missing} SNPs missing MAF in chr{chrnum}")

        df.to_csv(result_file, index=False)

print("Done.")