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

G = open_bed('data/fourier_ls-chr6-1167_train.bed')
train_geno = G.read(index=np.s_[:, :])

print(train_geno.shape)

G = open_bed('data/fourier_ls-chr6-1167_test.bed')
test_geno = G.read(index=np.s_[:, :])

print(test_geno.shape)

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

# data = load(file_name, data_dir)
# data = data.astype(np.int8)
train_data = train_geno.astype(np.int8)
valid_data = test_geno.astype(np.int8)
# print(data.shape)

# total_size = len(data)

# train_size = int(0.8 * total_size)
# valid_size = int(0.1 * total_size)
# test_size = total_size - train_size - valid_size

# indices = np.random.permutation(total_size)

# train_indices = indices[:train_size]
# valid_indices = indices[train_size:train_size + valid_size]
# test_indices = indices[train_size + valid_size:]

# train_data = data[train_indices]
# valid_data = data[valid_indices]
# test_data = data[test_indices]

train_data = torch.tensor(train_data, dtype=torch.long)
valid_data = torch.tensor(valid_data, dtype=torch.long)
# test_data = torch.tensor(test_data, dtype=torch.long)

print(train_data.shape)
print(valid_data.shape)
# print(test_data.shape)

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

ns = juice.structures.HCLT(
    train_data.float().to(device),
    num_latents = 1024,
    input_dist=dists.Categorical(num_cats=3)
)

pc = juice.compile(ns)
pc.to(device)

num_epochs = 400

optimizer = juice.optim.CircuitOptimizer(pc, lr = 0.05, pseudocount = 0.005, method = "EM")
scheduler = juice.optim.CircuitScheduler(
    optimizer,
    method = "multi_linear",
    lrs = [0.05, 0.01],
    milestone_steps = [0, len(train_loader) * num_epochs]
)

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

juice.save('circuits/ns_chr6-1167-1024.jpc', ns)
juice.save('circuits/pc_chr6-1167-1024.jpc', pc)

##########################################################################################################################################################################

# from torchmetrics import R2Score

# r2_metric = R2Score().to(device)

# # del data, train_data, valid_data
# # data = test_data
# data = torch.tensor(valid_data, dtype=torch.long).to(device)

# num_samples = data.size(0)
# num_features = data.size(1)

# # acc = np.zeros(num_features)
# r2s = np.zeros(num_features)
# for pos in range(num_features):
#     print(pos)
#     false_array = torch.full((num_features,), False, dtype=torch.bool).to(device)
#     # false_array[pos] = False
#     missing_mask = torch.tensor(false_array).to(device)

#     data0 = data.clone()
#     data0[:, pos] = 0
#     lls0 = juice.queries.marginal(pc, data=data0, missing_mask=missing_mask)

#     del data0

#     data1 = data.clone()
#     data1[:, pos] = 1
#     lls1 = juice.queries.marginal(pc, data=data1, missing_mask=missing_mask)

#     del data1

#     data2 = data.clone()
#     data2[:, pos] = 2
#     lls2 = juice.queries.marginal(pc, data=data2, missing_mask=missing_mask)

#     del data2

#     lls_stack = torch.stack([lls0, lls1, lls2], dim=-1)

#     predictions = torch.argmax(lls_stack, dim=-1).squeeze()
#     original = data[:, pos]

#     # correct_predictions = (predictions == original).sum().item()
#     # accuracy = correct_predictions / num_samples
#     r2_score = r2_metric(predictions, original)

#     # print(accuracy)
#     print(r2_score)

#     r2s[pos] = r2_score
#     # acc[pos] = accuracy
