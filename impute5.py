import numpy as np
import pandas as pd
import subprocess
import os
import math

def process_plink_data(plink_file_prefix):
    vcf_file = f"data/{plink_file_prefix}.vcf"
    # print("Exporting PLINK data to VCF...")
    export_command = [
        "../plink2", "--bfile", f"data/{plink_file_prefix}",
        "--recode", "vcf", "--out", f"data/{plink_file_prefix}",
        "--silent"
    ]
    
    if not os.path.exists(vcf_file):
        subprocess.run(export_command, check=True)

    bcftools_command = [
        "../bcftools/bcftools", "view", "-Oz", "-o", f"data/{plink_file_prefix}.bcf", vcf_file
    ]

    if not os.path.exists(f"data/{plink_file_prefix}.bcf"):
        subprocess.run(bcftools_command, check=True)

        # print("BCF conversion complete.")

        # print("Adding AC field..")
        subprocess.run([
        '../bcftools/bcftools', '+fill-tags', f'data/{plink_file_prefix}.bcf', '-Ob', '-o', f'data/{plink_file_prefix}_AC.bcf', '--', '-t', 'AN,AC'
        ], stdout=subprocess.DEVNULL)
        # print("Adding index file..")
        subprocess.run(['../bcftools/bcftools', 'index', f'data/{plink_file_prefix}_AC.bcf'])

    if not os.path.exists(f"data/{plink_file_prefix}_xcf.bcf"):
        subprocess.run([
            '../impute5_v1.2.0/xcftools_static', 'view',
            '-i', f"data/{plink_file_prefix}_AC.bcf",
            '-o', f"data/{plink_file_prefix}_xcf.bcf",
            '-O', 'sh',
            '-r', '6',
            '-T8',
            '-m', '0.03125'
        ])

def process_plink_data_with_drop(plink_file_prefix, rs_ids):
    vcf_file = f"data/{plink_file_prefix}.vcf"
    # print("Exporting PLINK data to VCF...")
    export_command = [
        "../plink2", "--bfile", f"data/{plink_file_prefix}",
        "--recode", "vcf", "--out", f"data/{plink_file_prefix}",
        "--silent"
    ]
    subprocess.run(export_command)

    modified_vcf_file = f"modified_test.vcf"
    with open(vcf_file, 'r') as f:
        lines = f.readlines()

    with open(modified_vcf_file, 'w') as out_file:
        for line in lines:
            if line.startswith('#'):
                out_file.write(line)
                continue
            
            fields = line.strip().split('\t')
            snp_id = fields[2]

            if snp_id in rs_ids:
                # fields[9:] = ['./.' for _ in fields[9:]]
                continue
                
            out_file.write('\t'.join(fields) + '\n')

    # print("VCF modification complete.")

    bcftools_command = [
        "../bcftools/bcftools", "view", "-Oz", '--threads', '32', "-o", "modified_test.bcf", modified_vcf_file
    ]
    subprocess.run(bcftools_command)

    # print("BCF conversion complete.")

    # print("Adding AC field..")
    subprocess.run([
     '../bcftools/bcftools', '+fill-tags', 'modified_test.bcf', '-Ob', '-o', 'modified_test_AC.bcf', '--', '-t', 'AN,AC'
    ], stdout=subprocess.DEVNULL)
    # print("Adding index file..")
    subprocess.run(['../bcftools/bcftools', 'index', 'modified_test_AC.bcf'])

    # print('Converting to XCF...')
    subprocess.run([
        '../impute5_v1.2.0/xcftools_static', 'view',
        '-i', 'modified_test_AC.bcf',
        '-o', "modified_test_xcf.bcf",
        '-O', 'sh',
        '-r', '6',
        '-T8',
        '-m', '0.03125'
    ])

def extract_imputed_genotype_array(vcf_file, snp_set, correct_genotype_array, num_samples):
    results = {}
    
    with open(vcf_file, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            if line.startswith('#'):
                continue  # Skip header lines
            
            fields = line.strip().split('\t')
            snp_id = f"{fields[0]}:{fields[1]}"  # Format: chr:pos
            
            if snp_id in snp_set:
                c = correct_genotype_array[snp_id]

                genotype_data = fields[9:]
                genotype_array = []
                expected_counts = []
                probs = np.zeros(num_samples)
                log_probs = np.zeros(num_samples)
                log_probs_filtered = np.zeros(num_samples)

                for i, genotype in enumerate(genotype_data):
                    gt_info, _, gp_info = genotype.split(':')
                    
                    # Convert genotype format to 0, 1, or 2
                    if gt_info == "0|0":
                        genotype_array.append(0)
                    elif gt_info == "0|1" or gt_info == "1|0":
                        genotype_array.append(1)
                    elif gt_info == "1|1":
                        genotype_array.append(2)
                    else:
                        genotype_array.append(-1)

                    prob_values = list(map(float, gp_info.split(',')))
                    prob_0 = prob_values[0]
                    prob_1 = prob_values[1]
                    prob_2 = prob_values[2]
                    expected_count = prob_0 * 0 + prob_1 * 1 + prob_2 * 2
                    expected_counts.append(expected_count)

                    correct = c[i]
                    probs[i] = prob_values[correct]
                    if prob_values[correct] != 0:
                        log_probs[i] = math.log(prob_values[correct])
                        log_probs_filtered[i] = math.log(prob_values[correct])
                    else:
                        prob_values = np.array(prob_values) + 0.0001
                        prob_values /= np.sum(prob_values)
                        log_probs[i] = math.log(prob_values[correct])
                        log_probs_filtered[i] = 1

                results[snp_id] = (genotype_array, expected_counts, probs, log_probs, log_probs_filtered)

    return results


def extract_test_genotype_array(vcf_file, snp_set):
    results = {}

    with open(vcf_file, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            if line.startswith('#'):
                continue  # Skip header lines
            
            fields = line.strip().split('\t')
            snp_id = f"{fields[0]}:{fields[1]}"  # Format: chr:pos
            
            if snp_id in snp_set:
                genotype_data = fields[9:]
                genotype_array = []

                for genotype in genotype_data:
                    if genotype == "0/0":
                        genotype_array.append(0)
                    elif genotype == "0/1" or genotype == "1/0":
                        genotype_array.append(1)
                    elif genotype == "1/1":
                        genotype_array.append(2)
                    else:
                        genotype_array.append(-1)

                results[snp_id] = genotype_array

    return results 

def compute_r_squared(array1, array2):
    array1 = np.array(array1)
    array2 = np.array(array2)
    
    mean_array1 = np.mean(array1)
    
    sse = np.sum((array1 - array2) ** 2)
    sst = np.sum((array1 - mean_array1) ** 2)

    r_squared = 1 - (sse / sst)
    
    return r_squared

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

def find_min_max_positions_bim(bim_file):
    with open(bim_file, 'r') as f:
        positions = [int(line.split()[3]) for line in f]
    
    min_pos = min(positions)
    max_pos = max(positions)
    
    return min_pos, max_pos

########################################################################################################################


bim_file = "data/fourier_ls-chr6-1167_train.bim"
idx1, idx2 = find_min_max_positions_bim(bim_file)

print(f"Min position: {idx1}")
print(f"Max position: {idx2}")

buffer_region = f"6:{idx1}-{idx2}"

os.environ['BCFTOOLS_PLUGINS'] = '/scratch2/prateek/bcftools/plugins'

plink_file_train_prefix = "fourier_ls-chr6-1167_train"
plink_file_test_prefix = "fourier_ls-chr6-1167_test"

process_plink_data(plink_file_train_prefix)

bim_file = f"data/{plink_file_test_prefix}.bim"
rs_ids = []

with open(bim_file, 'r') as f:
    for line in f:
        fields = line.split()
        rs_id = fields[1]
        rs_ids.append(rs_id)

num_snps = len(rs_ids)
fam_file = f"data/{plink_file_test_prefix}.fam"
num_samples = sum(1 for _ in open(fam_file))

r2s = np.zeros(num_snps)
pseudolikelihoods = np.zeros(num_samples)
pseudolikelihoods_filtered = np.zeros((num_samples, num_snps))

# for idx, rs_id in enumerate(rs_ids):
#     print(f"Dropping SNP: {rs_id} (#{idx+1})")
#     num = int(rs_id.split(':')[1])

batch_size = 1  # Define batch size for SNPs to drop in each iteration

joints = np.zeros((num_samples, (num_snps + batch_size - 1) // batch_size))

for batch_idx in range(0, len(rs_ids), batch_size):
    snp_set = rs_ids[batch_idx:batch_idx + batch_size]
    print(f"Dropping SNP set: {snp_set}")

    i = int(rs_ids[batch_idx].split(':')[1])
    j = int(rs_ids[min(batch_idx + batch_size, len(rs_ids)) - 1].split(':')[1])

    if i == j:
        j += 1
        if j == (idx2+1):
            i -= 1
            j -= 1

    process_plink_data_with_drop(plink_file_test_prefix, snp_set)
    
    result = subprocess.run([
        '../impute5_v1.2.0/impute5_v1.2.0_static',
        '--h', f'data/{plink_file_train_prefix}_xcf.bcf',
        '--m', '../b37_recombination_maps/chr6.b37.gmap.gz',
        '--g', 'modified_test_xcf.bcf',
        '--r', f"6:{i}-{j}",
        '--buffer-region', buffer_region,
        '--o', f"imputed_{batch_idx}.vcf",
        '--l', f"imputed_{batch_idx}.log",
        '--threads', '32'
    ])
    
    test_arrays = extract_test_genotype_array('data/fourier_ls-chr6-1167_test.vcf', snp_set)
    results = extract_imputed_genotype_array(f'imputed_{batch_idx}.vcf', snp_set, test_arrays, num_samples)

    for a, snp in enumerate(snp_set):
        test_genotype_array = test_arrays[snp]
        imputed_genotype_array = results[snp][0]
        expected_counts = results[snp][1]
        probs = results[snp][2]
        log_probs = results[snp][3]
        log_probs_filtered = results[snp][4]

        # print(test_genotype_array)
        # print(probs)
        # r2_prev = compute_r_squared(test_genotype_array, imputed_genotype_array)
        r2 = compute_r_squared(test_genotype_array, expected_counts)
        # print(r2_prev)
        print(r2)

        snp_index = batch_idx + a
        r2s[snp_index] = r2
        pseudolikelihoods += log_probs
        pseudolikelihoods_filtered[:, snp_index] = log_probs_filtered

        joints[:, batch_idx // batch_size] += log_probs

    # Delete the generated files after the SNP is processed
    files_to_delete = [f"imputed_{batch_idx}.log", f"imputed_{batch_idx}.vcf", "modified_test_AC.bcf", "modified_test_AC.bcf.csi", "modified_test_xcf.bcf", "modified_test_xcf.bcf.csi", "modified_test_xcf.bin", "modified_test_xcf.fam", "modified_test.bcf", "modified_test.vcf"]
    for file in files_to_delete:
        if os.path.exists(file):
            os.remove(file)
        else:
            print(f"File {file} not found for deletion")

    print(r2s)
    print(pseudolikelihoods)

df = pd.DataFrame(joints)
df.to_csv(f'results/joints_mask{batch_size}_chr6-1167_impute5.csv', index=False, header=False)

# df = pd.DataFrame(pseudolikelihoods_filtered)
# df.to_csv(f'results/pseudolikelihoods_mask{batch_size}_filtered_impute5.csv', index=False, header=False)
# np.savetxt(f'results/pseudolikelihoods_mask{batch_size}_chr6-1167_impute5', pseudolikelihoods)
np.savetxt(f'results/r2s_mask{batch_size}_chr6-1167_impute5_correct', r2s)

# def find_genotype_mismatches(truth_array, prediction_array, missing_value=-1):
    #     truth_array = np.array(truth_array)
    #     prediction_array = np.array(prediction_array)
        
    #     valid_positions = (truth_array != missing_value) & (prediction_array != missing_value)
        
    #     mismatch_indices = np.where((truth_array != prediction_array) & valid_positions)[0]
        
    #     for index in mismatch_indices:
    #         print(f"Mismatch at index {index}: Truth={truth_array[index]}, Prediction={prediction_array[index]}")
    #     print(len(mismatch_indices))
    #     print(np.sum(valid_positions))

    # find_genotype_mismatches(test_genotype_array, imputed_genotype_array)