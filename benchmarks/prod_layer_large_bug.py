"""Minimal unit test for the >2048-edge product kernel in PyJuice.

`ProdLayer._forward_backward` dispatches to `_forward_backward_kernel_large`
whenever a product node has more than 2048 children. That path is what an HCLT
enters when many SNPs are invariant: a monomorphic column has zero entropy, so
its mutual information with everything is zero, the Chow-Liu step is free to
attach all such variables to one node, and the tree collapses into a star whose
hub exceeds the 2048-edge threshold.

This builds that shape directly, with no data and no structure learning, so the
kernel can be exercised in seconds.

    python prod_layer_large_bug.py --vars 4096
"""

import argparse
import sys
import traceback

import torch

sys.setrecursionlimit(1000000)


def build(num_vars, device):
    import pyjuice as juice
    from pyjuice.nodes import multiply, summate, inputs, set_block_size
    from pyjuice.nodes.distributions import Categorical

    with set_block_size(block_size=1):
        leaves = [inputs(v, num_node_blocks=1, dist=Categorical(num_cats=2))
                  for v in range(num_vars)]
        prod = multiply(*leaves)          # one product node over every variable
        root = summate(prod, num_node_blocks=1, block_size=1)
    return juice.compile(root).to(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vars", type=int, nargs="+", default=[1024, 2048, 4096],
                   help="product fan-in to test; the large kernel engages above 2048")
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    device = torch.device("cuda")
    for nv in args.vars:
        print(f"--- product node with {nv} children "
              f"({'large' if nv > 2048 else 'standard'} kernel)", flush=True)
        try:
            pc = build(nv, device)
            x = torch.randint(0, 2, (args.batch, nv), device=device)
            lls = pc(x)
            torch.cuda.synchronize()
            print(f"    forward OK    LL={lls.mean().item():.4f}", flush=True)
            lls.mean().backward()
            torch.cuda.synchronize()
            print(f"    backward OK", flush=True)
            del pc
            torch.cuda.empty_cache()
        except Exception:
            print("    FAILED:")
            traceback.print_exc()
            break


if __name__ == "__main__":
    main()
