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

# G = open_bed('data/fourier_ls-chr6-1167_train.bed')
# train_geno = G.read(index=np.s_[:, :])

# print(train_geno.shape)

# G = open_bed('data/fourier_ls-chr6-1167_test.bed')
# test_geno = G.read(index=np.s_[:, :])

# print(test_geno.shape)

device = torch.device("cuda:0")
np.random.seed(1)

data_dir = "data1kg"
file_name = "10K.data"

def load(file_name, data_dir):
    filename = f"{data_dir}/{file_name}"
    dataframe = pd.read_csv(filename, header=None,
                            true_values=["1", "2", "3"],
                            false_values=["0"], dtype=object)
    data = dataframe.iloc[:, 0].str.split(' ')
    return np.array([np.array(entry) for entry in data])

data = load(file_name, data_dir)
data = data.astype(np.int8)
# train_data = train_geno.astype(np.int8)
# valid_data = test_geno.astype(np.int8)
print(data.shape)

total_size = len(data)

train_size = int(0.8 * total_size)
valid_size = total_size - train_size

indices = np.random.permutation(total_size)

train_indices = indices[:train_size]
valid_indices = indices[train_size:]

train_data = data[train_indices]
valid_data = data[valid_indices]

train_data = torch.tensor(train_data, dtype=torch.long)
valid_data = torch.tensor(valid_data, dtype=torch.long)

print(train_data.shape)
print(valid_data.shape)

train_loader = DataLoader(
    dataset = TensorDataset(train_data),
    batch_size = 1024,
    shuffle = True,
    drop_last = True
)
valid_loader = DataLoader(
    dataset = TensorDataset(valid_data),
    batch_size = 1024,
    shuffle = False,
    drop_last = True
)

latents = 8192

ns = juice.structures.HCLT(
    train_data.float().to(device),
    num_latents = latents,
    input_dist=dists.Categorical(num_cats=3)
)

pc = juice.compile(ns)
pc.to(device)

num_epochs = 400

for batch in train_loader:
    x = batch[0].to(device)

    lls = pc(x, record_cudagraph = True)
    lls.mean().backward()
    break

for epoch in range(1, num_epochs+1):
    t0 = time.time()

    # Manually zeroling out the flows
    pc.init_param_flows(flows_memory = 0.0)

    train_ll = 0.0
    for batch in train_loader:
        x = batch[0].to(device)

        # We only run the forward and the backward pass, and accumulate the flows throughout the epoch
        lls = pc(x)
        lls.mean().backward()

        train_ll += lls.mean().detach().cpu().numpy().item()

    # Set step size to 1.0 for full-batch EM
    pc.mini_batch_em(step_size = 1.0, pseudocount = 0.005)

    train_ll /= len(train_loader)

    t1 = time.time()
    test_ll = 0.0
    for batch in valid_loader:
        x = batch[0].to(pc.device)
        lls = pc(x)
        test_ll += lls.mean().detach().cpu().numpy().item()

    test_ll /= len(valid_loader)
    t2 = time.time()
    print(f"[Epoch {epoch}/{num_epochs}][train LL: {train_ll:.2f}; val LL: {test_ll:.2f}].....[train forward+backward+step {t1-t0:.2f}; val forward {t2-t1:.2f}] ")

juice.save(f'circuits/ns_10K-{latents}.jpc', ns)
juice.save(f'circuits/pc_10K-{latents}.jpc', pc)
