"""Do the paper's core results reproduce under a different PyJuice version?

Loads the trained GPC used in the paper and exercises the three things every
reported result depends on: the held-out log-likelihood, direct (conditional)
imputation, and sampling of artificial genomes. Run once per PyJuice checkout
and compare; the likelihood and the conditionals are deterministic and should
agree to numerical precision, while sampling is stochastic and is compared on
summary statistics rather than sample by sample.

    PYTHONPATH=/scratch2/prateek/pyjuice/src      python verify_pyjuice_versions.py --tag genetic-pc
    PYTHONPATH=/scratch2/prateek/pyjuice-main/src python verify_pyjuice_versions.py --tag main+fix
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.setrecursionlimit(1000000)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL = f"{ROOT}/results/1KG/8020/hclt/pc_10K_8020_nooverlap_4006-128_5000epochs_ps0.005.jpc"
TEST = f"{ROOT}/results/1KG/8020/data/8020_test.txt"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "version_check")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True, help="label for this PyJuice checkout")
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--num-snps", type=int, default=50, help="SNPs to impute directly")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import pyjuice as juice
    print(f"[{args.tag}] pyjuice from {os.path.dirname(juice.__file__)}", flush=True)

    device = torch.device("cuda")
    pc = juice.compile(juice.load(MODEL)).to(device)
    test = torch.tensor(np.loadtxt(TEST, dtype=np.int8, delimiter=' '), dtype=torch.long)
    res = dict(tag=args.tag, pyjuice_path=os.path.dirname(juice.__file__),
               num_vars=int(pc.num_vars), test_shape=list(test.shape))

    # 1. held-out log-likelihood, the number reported in Table 1
    with torch.no_grad():
        lls = [pc(test[s:s + 256].to(device)).mean().item() for s in range(0, test.shape[0], 256)]
    res["test_ll"] = float(np.mean(lls))
    print(f"  held-out LL                {res['test_ll']:.6f}", flush=True)

    # 2. direct imputation: leave one SNP out and read its conditional
    torch.manual_seed(args.seed)
    x = test[:512].to(device)
    mask = torch.zeros(pc.num_vars, dtype=torch.bool, device=device)
    dos, truth = [], []
    for pos in range(0, args.num_snps):
        mask[pos] = True
        with torch.no_grad():
            probs = juice.queries.conditional(pc, data=x, missing_mask=mask)
        dos.append(probs[:, pos, 1].cpu().numpy())
        truth.append(x[:, pos].cpu().numpy())
        mask[pos] = False
    dos, truth = np.array(dos), np.array(truth)
    r2 = [np.corrcoef(t, d)[0, 1] ** 2 if t.std() > 0 else np.nan
          for t, d in zip(truth, dos)]
    res["imputation_mean_r2"] = float(np.nanmean(r2))
    res["imputation_dosage_sum"] = float(dos.sum())
    print(f"  direct imputation mean r2  {res['imputation_mean_r2']:.6f}", flush=True)
    print(f"  dosage checksum            {res['imputation_dosage_sum']:.4f}", flush=True)

    # 3. sampling: stochastic, so compare distributional summaries
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    s = juice.queries.sample(pc, num_samples=args.num_samples).cpu().numpy()
    real_af = test.numpy().mean(0)
    syn_af = s.mean(0)
    res["sample_shape"] = list(s.shape)
    res["sample_mean_af"] = float(syn_af.mean())
    res["af_corr_with_real"] = float(np.corrcoef(syn_af, real_af)[0, 1])
    print(f"  AG mean allele frequency   {res['sample_mean_af']:.6f}", flush=True)
    print(f"  AG vs real AF correlation  {res['af_corr_with_real']:.6f}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{args.tag}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
