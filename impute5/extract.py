# %%
import os
import re
import pandas as pd
import numpy as np
# %%
for method in ["10K_real_8020", "10K_hclt_8020", "10K_rbm_8020", "10K_wgan_8020"]:

    # Directory containing the files
    directory = f"out/impute_{method}"

    # List to hold parsed data
    data = []
    present = []
    i = 1
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # print(i)
        filepath = os.path.join(directory, filename)
        
        if os.path.isfile(filepath):
            with open(filepath, "r") as file:
                lines = file.readlines()
            
            # Extract SNP set
            snp_set = None
            r2_values = []

            for line in lines:
                # Find the SNP set
                if "Dropping SNP set:" in line:
                    match = re.search(r"Dropping SNP set: \[(.*?)\]", line)
                    if match:
                        snp_set = match.group(1)
                
                # Find R² values (assuming they are floating-point numbers)
                try:
                    value = float(line.strip())
                    r2_values.append(value)
                except ValueError:
                    continue
            
            # Extract the last two R² values
            if len(r2_values) >= 2 and snp_set not in present:
                present.append(snp_set)
                r2_last, r2_second_last = r2_values[-2], r2_values[-1]
                data.append({
                "SNP Set": snp_set,
                "R2": r2_last,
                "R2 Geno": r2_second_last
            })
            else:
                os.remove(filepath)
        i += 1

    df = pd.DataFrame(data)
    df.to_csv(f"results/bootstrap/{method}_correct.csv", index=False)
    # print(df)

    true_snps_file = "../aux/10K_SNP.legend"
    true_snps = pd.read_csv(true_snps_file, sep=" ")

    # Extract SNP positions from the true file
    true_positions = set(true_snps["position"].astype(int))

    # Extract SNP positions from the present SNPs
    present_positions = set()
    for snp_set in present:
        if snp_set:  # Skip None or empty values
            # Parse SNP positions from the set
            positions = re.findall(r"\d+:(\d+)", snp_set)
            present_positions.update(map(int, positions))

    # Identify missing SNPs
    missing_positions = true_positions - present_positions

    # Filter missing SNPs from the true file
    missing_snps = true_snps[true_snps["position"].isin(missing_positions)]

    # Save missing SNPs to a file
    output_missing_snp_file = f"rerun/bootstrap/impute_{method}_correct.txt"

    # Format the missing SNP positions
    missing_snp_positions = [
        f"'{snp.split('_')[0]}'"  # Extract chromosome and position, ignoring alleles
        for snp in missing_snps["id"]
    ]

    # Write to the file
    with open(output_missing_snp_file, "w") as file:
        for position in missing_snp_positions:
            file.write(f"{position}\n")

    print(len(missing_snps))
    print(f"Missing SNP positions saved to: {output_missing_snp_file}")

    missing_snps_file = f"rerun/bootstrap/impute_{method}_correct.txt"
    mapping_file = f"rerun/bootstrap/impute_{method}_correct_map.txt"

    # Read the true SNP positions
    true_positions = []
    with open(true_snps_file, "r") as f:
        for line in f.readlines()[1:]:  # Skip header if present
            parts = line.split()
            true_positions.append(parts[1])  # Extract the position column

    # Read the missing SNP positions
    missing_positions = []
    with open(missing_snps_file, "r") as f:
        for line in f.read().splitlines():
            pos = line.strip("'").split(":")[1]  # Extract the position from "chr:pos"
            missing_positions.append(pos)

    # Create a mapping of indices
    with open(mapping_file, "w") as f:
        for missing_pos in missing_positions:
            try:
                index = true_positions.index(missing_pos)
                f.write(f"{index+1}\n")
            except ValueError:
                # Log SNPs not found in the true SNPs file
                print(f"Warning: Position {missing_pos} not found in true SNPs")

# %%
