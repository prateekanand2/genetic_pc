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

# train_loader = DataLoader(
#     dataset = TensorDataset(train_data),
#     batch_size = 1024,
#     shuffle = True,
#     drop_last = True
# )
# valid_loader = DataLoader(
#     dataset = TensorDataset(valid_data),
#     batch_size = 1024,
#     shuffle = False,
#     drop_last = True
# )

ns = juice.load('circuits/pc_chr6-1167-1024.jpc')
pc = juice.compile(ns)
pc.to(device)

batch_size = 512
dataset = TensorDataset(valid_data)
dataloader = DataLoader(dataset, batch_size=batch_size)

num_samples = valid_data.size(0)
num_features = valid_data.size(1)

sse_acc = np.zeros(num_features)
sst_acc = np.zeros(num_features)

pseudolikelihoods = np.zeros(num_samples)

means = torch.mean(valid_data.float(), dim=0).to(device)

num_masked = 1

joints = np.zeros((num_samples, (num_features + num_masked - 1) // num_masked))

i = 1
batch_start = 0
for batch in dataloader:
    print(f'Batch {i}')
    data = batch[0].to(device)
    batch_size = data.size(0)

    batch_end = batch_start + batch_size

    for pos in range(0, num_features, num_masked):
        print(pos)
        end_pos = min(pos + num_masked, num_features)

        false_array = torch.full((num_features,), False, dtype=torch.bool).to(device)
        false_array[pos:end_pos] = True
        missing_mask = torch.tensor(false_array).to(device)

        lls = juice.queries.conditional(pc, data=data, missing_mask=missing_mask)

        probs = lls[:, pos:end_pos, :]

        # print(probs)
        # print(probs.shape)

        for idx, feature_pos in enumerate(range(pos, end_pos)):
            original = data[:, feature_pos]

            prob_0 = probs[:, idx, 0]
            prob_1 = probs[:, idx, 1]
            prob_2 = probs[:, idx, 2]
            expected_counts = prob_0 * 0 + prob_1 * 1 + prob_2 * 2

            # max_probs, predictions = torch.max(probs[:, idx:idx+1, :], dim=-1)
            # predictions = predictions.squeeze()

            correct_probs = torch.gather(probs[:, idx].squeeze(1), 1, original.unsqueeze(1)).squeeze()
            log_correct_probs = torch.log(correct_probs)

            pseudolikelihoods[batch_start:batch_end] += log_correct_probs.cpu().numpy()
            joints[batch_start:batch_end, pos // num_masked] += log_correct_probs.cpu().numpy()

            # Calculate the sum of squared errors (SSE) for the current batch
            sse_batch = torch.sum((original - expected_counts) ** 2).item()

            # Calculate the total sum of squares (SST) for the current batch
            sst_batch = torch.sum((original - means[feature_pos]) ** 2).item()

            # Accumulate SSE and SST for the current feature
            sse_acc[feature_pos] += sse_batch
            sst_acc[feature_pos] += sst_batch

    batch_start = batch_end
    i += 1
    
# After processing all batches, calculate R^2 for each feature

r2s = 1 - (sse_acc / sst_acc)

print("R2 scores:", r2s)
print("Pseudolikelihoods:", pseudolikelihoods)

df = pd.DataFrame(joints)
df.to_csv(f'results/joints_mask{num_masked}_chr6-1167-1024.csv', index=False, header=False)

# np.savetxt('results/r2s_chr6-1167-1024_correct', r2s)
np.savetxt(f'results/r2s_mask{num_masked}_chr6-1167-1024_correct', r2s)
# np.savetxt(f'results/pseudolikelihoods_mask{num_masked}_chr6-1167-1024', pseudolikelihoods)