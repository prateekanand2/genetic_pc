import pyjuice as juice
import torch
# import torchvision
import time
from torch.utils.data import TensorDataset, DataLoader
import pyjuice.nodes.distributions as dists
import numpy as np
import pandas as pd

import seaborn as sns
import pandas as pd
import numpy as np
import importlib
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from bed_reader import open_bed
import gc

split = "8020"
snps = f"1KG_{split}_4006"
latents = 128
ps = 0.005
num_epochs = 5000

print("Number of CUDA devices:", torch.cuda.device_count())
print(torch.version.cuda)

device = torch.device("cuda:0")
np.random.seed(1)

print(device)
print(os.getenv("TRITON_CACHE_DIR"))

ns = juice.load(f'/scratch2/prateek/genetic_pc/reproduce_final/UKBB/{split}/hclt/pc_{snps}-{latents}_{num_epochs}epochs_ps{ps}.jpc')
pc = juice.compile(ns)
pc.to(device)

print(ns.num_parameters())

samples = []
for i in range(50):
    print(i)
    s = juice.queries.sample(pc, num_samples = 100)
    for x in s.cpu():
        samples.append(x)

s = juice.queries.sample(pc, num_samples = 8)
for x in s.cpu():
    samples.append(x)

np_arrays = [tensor.numpy() for tensor in samples]
d = np.vstack(np_arrays)

np.savetxt(f'../../reproduce_final/UKBB/{split}/hclt/{snps}_SNP_HCLT_AG_{latents}_{num_epochs}epochs_ps{ps}.txt', d, fmt='%d')