import numpy as np
import argparse
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # Import tqdm for progress bar
import torch

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

def ComputeAATS(X,fake_X,s_X):
    CONCAT = torch.cat((X[:,:s_X],fake_X[:,:s_X]),1)
    dAB = torch.cdist(CONCAT.t(),CONCAT.t())    
    torch.diagonal(dAB).fill_(float('inf'))
    dAB = dAB.cpu().numpy()

    # the next line is use to tranform the matrix into
    #  d_TT d_TF   INTO d_TF- d_TT-  where the minus indicate a reverse order of the columns
    #  d_FT d_FF        d_FT  d_FF
    dAB[:int(dAB.shape[0]/2),:] = dAB[:int(dAB.shape[0]/2),::-1] 

    closest = dAB.argmin(axis=1) 
    for i in range(closest.shape[0]):
        s = (len(np.where(dAB[i,:]==closest[i])))
        if s>1:
            print('multiple min=',s)
    n = int(closest.shape[0]/2)

    ninv = 1/n
    correctly_classified = closest>=n  
    AAtruth = (closest[:n] >= n).sum()*ninv  # for a true sample, proba that the closest is in the set of true samples
    AAsyn = (closest[n:] >= n).sum()*ninv  # for a fake sample, proba that the closest is in the set of fake samples

    return AAtruth, AAsyn

def main():
    parser = argparse.ArgumentParser(description="Compute the Adversarial Accuracy Two-Sample Test (AATS)")
    parser.add_argument("--samples_path", type=str, required=True, help="Path to the synthetic samples file")
    args = parser.parse_args()

    num = 2504

    samples = np.loadtxt(args.samples_path)
    samples1 = samples[:num]
    samples2 = samples[num:]

    real = np.loadtxt('../../results/1KG/aats/data/aats_train.txt', dtype=int)
    scaler = StandardScaler()
    real = scaler.fit_transform(real)
    samples1 = scaler.transform(samples1)
    aa_truth, aa_syn = ComputeAATS(torch.tensor(real).T, torch.tensor(samples1).T, num)

    print("Train")
    print(f"AA Truth: {aa_truth}")
    print(f"AA Synth: {aa_syn}")
    print(f"AA TS: {(aa_truth + aa_syn) / 2}")

    real = np.loadtxt('../../results/1KG/aats/data/aats_test.txt', dtype=int)
    scaler = StandardScaler()
    real = scaler.fit_transform(real)
    samples2 = scaler.transform(samples2)
    aa_truth, aa_syn = ComputeAATS(torch.tensor(real).T, torch.tensor(samples2).T, num)

    
    print("Test")
    print(f"AA Truth: {aa_truth}")
    print(f"AA Synth: {aa_syn}")
    print(f"AA TS: {(aa_truth + aa_syn) / 2}")

if __name__ == "__main__":
    main()