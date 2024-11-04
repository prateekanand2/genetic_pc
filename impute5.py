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

def extract_imputed_genotype_array(vcf_file, target_snp, correct_genotype_array, num_samples):
    with open(vcf_file, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            if line.startswith('#'):
                continue  # Skip header lines
            
            fields = line.strip().split('\t')
            snp_id = f"{fields[0]}:{fields[1]}"  # Format: chr:pos
            
            if snp_id == target_snp:
                genotype_data = fields[9:]  # Genotype fields for all samples
                genotype_array = []
                log_probs = np.zeros(num_samples)
                log_probs_filtered = np.zeros(num_samples)

                # Parse genotype data
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
                        genotype_array.append(-1)  # for missing or unknown genotypes

                    prob_values = list(map(float, gp_info.split(',')))
                    correct = correct_genotype_array[i]
                    if prob_values[correct] != 0:
                        log_probs[i] = math.log(prob_values[correct])
                        log_probs_filtered[i] = math.log(prob_values[correct])
                    else:
                        prob_values = np.array(prob_values) + 0.0001
                        prob_values /= (np.sum(prob_values))
                        log_probs[i] = math.log(prob_values[correct])
                        log_probs_filtered[i] = 1
                
                return genotype_array, log_probs, log_probs_filtered
    
    return None, None  # SNP not found in the VCF file


def extract_test_genotype_array(vcf_file, target_snp):

    with open(vcf_file, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            if line.startswith('#'):
                continue  # Skip header lines
            
            fields = line.strip().split('\t')
            snp_id = f"{fields[0]}:{fields[1]}"  # Format: chr:pos
            
            if snp_id == target_snp:
                genotype_data = fields[9:]  # Genotype fields for all samples
                genotype_array = []

                # Parse genotype data
                for genotype in genotype_data:
                    # Convert genotype format to 0, 1, or 2
                    if genotype == "0/0":
                        genotype_array.append(0)
                    elif genotype == "0/1" or genotype == "1/0":
                        genotype_array.append(1)
                    elif genotype == "1/1":
                        genotype_array.append(2)
                    else:
                        genotype_array.append(-1)  # for missing or unknown genotypes
                
                return genotype_array
    
    return None  # SNP not found in the VCF file

def compute_r_squared(array1, array2):
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

for idx, rs_id in enumerate(rs_ids):
    print(f"Dropping SNP: {rs_id} (#{idx+1})")
    num = int(rs_id.split(':')[1])

    i = num
    j = num + 1
    if num == idx2:
        i = idx2 - 1
        j = idx2

    process_plink_data_with_drop(plink_file_test_prefix, rs_id)
    
    result = subprocess.run([
        '../impute5_v1.2.0/impute5_v1.2.0_static',
        '--h', f'data/{plink_file_train_prefix}_xcf.bcf',
        '--m', '../b37_recombination_maps/chr6.b37.gmap.gz',
        '--g', 'modified_test_xcf.bcf',
        '--r', f"6:{i}-{j}",
        '--buffer-region', buffer_region,
        '--o', f"imputed_{idx}.vcf",
        '--l', f"imputed_{idx}.log",
        '--threads', '32'
    ])
    
    test_genotype_array = extract_test_genotype_array('data/fourier_ls-chr6-1167_test.vcf', rs_id)
    imputed_genotype_array, log_probs, log_probs_filtered = extract_imputed_genotype_array(f'imputed_{idx}.vcf', rs_id, test_genotype_array, num_samples)

    r2 = compute_r_squared(imputed_genotype_array, test_genotype_array)
    print(r2)

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

    r2s[idx] = r2
    pseudolikelihoods += log_probs
    pseudolikelihoods_filtered[:, idx] = log_probs_filtered

    # Delete the generated files after the SNP is processed
    files_to_delete = [f"imputed_{idx}.log", f"imputed_{idx}.vcf", "modified_test_AC.bcf", "modified_test_AC.bcf.csi", "modified_test_xcf.bcf", "modified_test_xcf.bcf.csi", "modified_test_xcf.bin", "modified_test_xcf.fam", "modified_test.bcf", "modified_test.vcf"]
    for file in files_to_delete:
        if os.path.exists(file):
            os.remove(file)
        else:
            print(f"File {file} not found for deletion")

# df = pd.DataFrame(pseudolikelihoods_filtered)
# df.to_csv('results/pseudolikelihoods_filtered_impute5.csv', index=False, header=False)
np.savetxt('results/pseudolikelihoods_chr6-1167_impute5', pseudolikelihoods)
# np.savetxt('results/r2s_chr6-1167_impute5', r2s)
