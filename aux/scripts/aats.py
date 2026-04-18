import numpy as np
import argparse
import torch

def _l1_matrix(A, B):
    """
    Exact L1/cityblock distance matrix between rows of A and B.
    For binary (0/1) data this equals Hamming distance and is computed
    via matrix multiplication: L1(a,b) = sum(a) + sum(b) - 2*(a·b),
    which is much faster than a pairwise loop.
    For float data it falls back to torch.cdist(p=1).
    Returns a (n_A, n_B) CPU tensor.
    """
    if A.is_floating_point():
        return torch.cdist(A, B, p=1).cpu()
    # binary integer path: cast to float just for matmul
    Af = A.float()
    Bf = B.float()
    return (Af.sum(1, keepdim=True) + Bf.sum(1, keepdim=True).T - 2 * (Af @ Bf.T)).cpu()

def _l2_matrix(A, B):
    """
    Euclidean (L2) distance matrix between rows of A and B.
    Returns a (n_A, n_B) CPU tensor.
    """
    return torch.cdist(A.float(), B.float(), p=2).cpu()

def ComputeAATS(real, syn, device='cpu', metric='cityblock', standardize=False, mu=None, std=None):
    """
    Compute AATS, matching the reference implementation from the
    artificial_genomes repo (computeAATS.py).

    Args:
        real:        (n, p) numpy array of real haplotypes
        syn:         (n, p) numpy array of synthetic haplotypes
        device:      torch device string ('cuda', 'cpu', etc.)
        metric:      'cityblock' (L1/Hamming) or 'euclidean' (L2)
        standardize: if True, z-score each feature before computing distances
        mu:          (p,) pre-computed mean for standardization; if None and
                     standardize=True, computed from real
        std:         (p,) pre-computed std for standardization; if None and
                     standardize=True, computed from real

    Returns:
        aa_truth, aa_syn
    """
    dist_fn = _l2_matrix if metric == 'euclidean' else _l1_matrix

    real = real.astype(float)
    syn  = syn.astype(float)
    if standardize:
        if mu is None:
            mu = real.mean(axis=0)
        if std is None:
            std = real.std(axis=0)
            std[std == 0] = 1.0
        real = (real - mu) / std
        syn  = (syn  - mu) / std

    real_t = torch.from_numpy(real).to(device)
    syn_t  = torch.from_numpy(syn).to(device)

    dTT_mat = dist_fn(real_t, real_t)
    dTT_mat.fill_diagonal_(float('inf'))
    dTT = dTT_mat.min(dim=1).values

    dTS_mat = dist_fn(real_t, syn_t)
    dTS = dTS_mat.min(dim=1).values  # nearest syn for each real point
    dST = dTS_mat.min(dim=0).values  # nearest real for each syn point

    dSS_mat = dist_fn(syn_t, syn_t)
    dSS_mat.fill_diagonal_(float('inf'))
    dSS = dSS_mat.min(dim=1).values

    aa_truth = (dTS > dTT).float().mean().item()
    aa_syn   = (dST > dSS).float().mean().item()

    return aa_truth, aa_syn

BASE = '/scratch2/prateek/genetic_pc_github/results'

DATASET_CONFIGS = {
    '1KG': {
        'num_syn':    2504,
        'train_path': f'{BASE}/1KG/aats/data/aats_train.txt',
        'test_path':  f'{BASE}/1KG/aats/data/aats_test.txt',
        'models': {
            'HCLT': f'{BASE}/1KG/aats/hclt/10K_hclt_aats_samples.txt',
            'RBM':  f'{BASE}/1KG/aats/rbm/10K_rbm_aats_samples.txt',
            'WGAN': f'{BASE}/1KG/aats/wgan/10K_wgan_aats_samples.txt',
        },
    },
    'UKBB': {
        'num_syn':    5000,
        'train_path': f'{BASE}/UKBB/aats/data/aats_train.txt',
        'test_path':  f'{BASE}/UKBB/aats/data/aats_test.txt',
        'models': {
            'HCLT': f'{BASE}/UKBB/aats/hclt/UKBB_hclt_aats_samples.txt',
            'RBM':  f'{BASE}/UKBB/aats/rbm/UKBB_rbm_aats_samples.txt',
            'WGAN': f'{BASE}/UKBB/aats/wgan/UKBB_wgan_aats_samples.txt',
        },
    },
}

def main():
    parser = argparse.ArgumentParser(description="Compute the Adversarial Accuracy Two-Sample Test (AATS)")
    parser.add_argument("--dataset", type=str, default='1KG', choices=['1KG', 'UKBB'],
                        help="Dataset to evaluate against (default: 1KG)")
    parser.add_argument("--metric", type=str, default='euclidean', choices=['cityblock', 'euclidean'],
                        help="Distance metric: 'euclidean' (L2) or 'cityblock' (L1/Hamming) (default: euclidean)")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Torch device (default: cuda if available, else cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subsampling real haplotypes (default: 42)")
    args = parser.parse_args()

    print(f"Dataset: {args.dataset}  |  Metric: {args.metric}  |  Device: {args.device}  |  Seed: {args.seed}")

    cfg = DATASET_CONFIGS[args.dataset]
    num = cfg['num_syn']
    rng = np.random.default_rng(args.seed)

    # Subsample real splits once — same rows used for every model
    real_splits = {}
    for label, path in [('Train', cfg['train_path']), ('Test', cfg['test_path'])]:
        real = np.loadtxt(path, dtype=int)
        if len(real) > num:
            idx = rng.choice(len(real), size=num, replace=False)
            real = real[idx]
        real_splits[label] = real

    # Fit standardization on R_tr only — same mu/std applied to all splits
    mu  = real_splits['Train'].astype(float).mean(axis=0)
    std = real_splits['Train'].astype(float).std(axis=0)
    std[std == 0] = 1.0

    # Collect results: (model, split, aa_truth, aa_syn, aa_truth_std, aa_syn_std)
    rows = []
    for model, samples_path in cfg['models'].items():
        print(f"  Computing {model}...", flush=True)
        samples   = np.loadtxt(samples_path)
        syn_train = samples[:num]
        syn_test  = samples[num:2 * num]

        for label, syn in [('Train', syn_train), ('Test', syn_test)]:
            real = real_splits[label]
            at,  as_  = ComputeAATS(real, syn, device=args.device, metric=args.metric, standardize=False)
            at_s, as_s = ComputeAATS(real, syn, device=args.device, metric=args.metric, standardize=True, mu=mu, std=std)
            rows.append((model, label, at, as_, at_s, as_s))

    # Print table
    col_w = 9
    header = (f"{'Model':<6} {'Split':<6}"
              f"  {'AA-Truth':>{col_w}} {'AA-Syn':>{col_w}}"
              f"  {'AA-Truth*':>{col_w}} {'AA-Syn*':>{col_w}}")
    sep = '-' * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    prev_model = None
    for model, split, at, as_, at_s, as_s in rows:
        if prev_model and model != prev_model:
            print()
        print(f"{model:<6} {split:<6}"
              f"  {at:>{col_w}.4f} {as_:>{col_w}.4f}"
              f"  {at_s:>{col_w}.4f} {as_s:>{col_w}.4f}")
        prev_model = model
    print(sep)
    print("* standardized (z-score by real data mean/std)")

if __name__ == "__main__":
    main()
