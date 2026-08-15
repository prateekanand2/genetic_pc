"""A non-homogeneous HMM in plain PyTorch, independent of pyjuice.

This exists to answer one question: is the chain we fit inside the circuit
framework the same model, fit to the same quality, as an HMM someone would write
directly? It implements the identical model, a first-order HMM with a
position-specific transition matrix and a position-specific emission table,

    P(x) = sum_z gamma(z_1) prod_t A_t(z_t, z_{t+1}) prod_t B_t(z_t, x_t),

and fits it with the identical algorithm, Baum-Welch with a pseudocount.

The E step uses the standard scaled forward recursion. Expected sufficient
statistics come from the identity that for an exponential family the gradient of
the log-likelihood with respect to a log-parameter is the expected count of the
corresponding event, so backpropagating the scaled forward log-likelihood
through the log-parameters yields exactly the Baum-Welch counts without writing
the backward recursion separately. The M step then renormalises those counts.

Because it is a loop over positions rather than a recursive graph traversal,
sequence length is bounded only by memory: it fits the full region in one model,
with no chunking.

    python hmm_reference.py --snps 2500 --epochs 5000     # matches 1KG chunk 0
    python hmm_reference.py --snps 10000 --epochs 5000    # the unchunked model
"""

import argparse
import math
import time

import numpy as np
import torch

PS = 0.005
REF_BATCH = 256


class NonHomogeneousHMM:
    def __init__(self, seq_length, num_latents, num_emits, device, seed=0, perturbation=4.0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.T, self.K, self.E = seq_length, num_latents, num_emits
        self.device = device

        def rand_log(*shape):
            # Match the circuit library's initialisation exactly: exp(-U*perturbation)
            # with U uniform on [0,1] and perturbation 4.0, then normalised. This
            # spans about a 55-fold range, far more dispersed than a plain uniform
            # draw. The dispersion matters: EM started near the uniform point sits
            # close to a symmetric saddle where all latent states are almost
            # interchangeable, and it converges to a markedly worse optimum.
            p = torch.exp(torch.rand(*shape, generator=g) * -perturbation)
            p = p / p.sum(-1, keepdim=True)
            return p.log().to(device).requires_grad_(True)

        self.log_init = rand_log(num_latents)
        self.log_A = rand_log(seq_length - 1, num_latents, num_latents)
        self.log_B = rand_log(seq_length, num_latents, num_emits)

    def params(self):
        return [self.log_init, self.log_A, self.log_B]

    def log_likelihood(self, x):
        """Mean scaled-forward log-likelihood over a batch of haplotypes.

        The recursion runs in probability space with alpha renormalised at every
        position, which is the standard scaling that keeps an HMM stable over
        long sequences. Working this way also keeps each step a (B,K) by (K,K)
        matmul, so autograd stores O(B*K) per position rather than O(B*K*K).
        """
        B = x.shape[0]
        A = self.log_A.exp()                                          # (T-1, K, K)
        Bm = self.log_B.exp()                                         # (T, K, E)
        emit = Bm.gather(2, x.t().unsqueeze(1).expand(self.T, self.K, B))  # (T, K, B)

        alpha = self.log_init.exp().unsqueeze(0) * emit[0].t()        # (B, K)
        c = alpha.sum(1)
        total = c.log()
        alpha = alpha / c.unsqueeze(1)
        for t in range(1, self.T):
            alpha = (alpha @ A[t - 1]) * emit[t].t()
            c = alpha.sum(1)
            total = total + c.log()
            alpha = alpha / c.unsqueeze(1)
        return total.mean()

    @torch.no_grad()
    def m_step(self, pseudocount):
        """Renormalise the accumulated expected counts.

        The pseudocount is a total prior mass spread evenly over the n outcomes
        of each distribution, so each entry receives ``pseudocount / n`` and each
        normaliser receives ``pseudocount``. This is a Dirichlet prior whose
        strength does not grow with fan-out, and it matches the convention the
        circuit library uses, where the same ``pseudocount / num_children`` and
        ``+ pseudocount`` appear in its EM kernels. Adding the raw pseudocount to
        every entry instead would put 128 times more prior mass on a 128-way
        transition than on a binary emission, which is enough to wash out the
        data and drive the chain towards independence.
        """
        for p in self.params():
            counts = p.grad
            n = counts.shape[-1]
            new_p = (counts + pseudocount / n) / (counts.sum(-1, keepdim=True) + pseudocount)
            p.copy_(new_p.log())
            p.grad = None


def evaluate(model, data, batch=REF_BATCH):
    with torch.no_grad():
        n = data.shape[0]
        nb = (n + batch - 1) // batch
        acc = 0.0
        for s in range(0, n, batch):
            acc += model.log_likelihood(data[s:s + batch].long()).item()
        return acc / nb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default="../results/1KG/8020/data/8020_train.txt")
    p.add_argument("--valid", default="../results/1KG/8020/data/8020_test.txt")
    p.add_argument("--snps", type=int, default=2500)
    p.add_argument("--offset", type=int, default=0, help="first SNP of the block")
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--latents", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--perturbation", type=float, default=4.0,
                   help="initialisation spread, matching the circuit library's default")
    p.add_argument("--report-every", type=int, default=100)
    args = p.parse_args()

    device = torch.device("cuda")
    lo, hi = args.offset, args.offset + args.snps
    tr = torch.from_numpy(np.loadtxt(args.train, dtype=np.int8, delimiter=' ')[:, lo:hi]).to(device)
    va = torch.from_numpy(np.loadtxt(args.valid, dtype=np.int8, delimiter=' ')[:, lo:hi]).to(device)
    n = tr.shape[0]
    print(f"train {tuple(tr.shape)}  valid {tuple(va.shape)}  "
          f"{args.latents} latents, {args.epochs} epochs", flush=True)

    model = NonHomogeneousHMM(args.snps, args.latents, 2, device, seed=args.seed,
                              perturbation=args.perturbation)

    nb = max(1, math.ceil(n / args.batch_size))
    bounds = [(i * n) // nb for i in range(nb + 1)]
    eff = n / nb
    # Same convention as the circuit runs: the pseudocount is defined at bs=256
    # and the per-batch mean makes the accumulated counts scale with nb.
    pseudocount = PS * (REF_BATCH / eff)
    print(f"  batch {args.batch_size} -> {nb} batches of ~{eff:.0f}, "
          f"pseudocount {pseudocount:g}", flush=True)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n, device=device)
        for i in range(nb):
            model.log_likelihood(tr[perm[bounds[i]:bounds[i + 1]]].long()).backward()
        model.m_step(pseudocount)

        if epoch % args.report_every == 0 or epoch == args.epochs:
            print(f"[Epoch {epoch}/{args.epochs}][train LL: {evaluate(model, tr):.2f}; "
                  f"val LL: {evaluate(model, va):.2f}]"
                  f".....[{(time.time() - t0) / 60:.1f} min]", flush=True)


if __name__ == "__main__":
    main()
