import os
import sys
import time
import torch
import numpy as np
import pyjuice as juice
from torch.utils.data import TensorDataset, DataLoader

# --- CONFIGURATION ---
class CFG:
    # Paths
    train_path = "../results/1KG/8020/data/8020_train.txt"
    valid_path = "../results/1KG/8020/data/8020_test.txt"
    output_prefix = "1kg_8020_hmm"
    
    # Model Hyperparameters
    latents = 128
    ps = 0.005
    num_epochs = 5000
    batch_size = 256
    
    # Execution Mode
    # Set num_chunks = 1 to train on the whole region
    # Set num_chunks = 4 (or more) to split the features
    num_chunks = 4
    
    # System
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 1

# Setup
sys.setrecursionlimit(100000)
torch.manual_seed(CFG.seed)
np.random.seed(CFG.seed)

def load_data():
    print(f"Loading data from {CFG.train_path}...")
    train_raw = np.loadtxt(CFG.train_path, dtype=np.int8, delimiter=' ')
    valid_raw = np.loadtxt(CFG.valid_path, dtype=np.int8, delimiter=' ')
    
    train_data = torch.tensor(train_raw, dtype=torch.long)
    valid_data = torch.tensor(valid_raw, dtype=torch.long)
    
    print(f"Train shape: {train_data.shape} | Valid shape: {valid_data.shape}")
    return train_data, valid_data

def train_model(train_subset, valid_subset, chunk_id):
    seq_length = train_subset.shape[1]
    tag = f"chunk_{chunk_id}" if CFG.num_chunks > 1 else "full_region"
    log_filename = f"{CFG.output_prefix}_{tag}_lat{CFG.latents}_ps{CFG.ps}.log"
    
    train_loader = DataLoader(TensorDataset(train_subset), batch_size=CFG.batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(valid_subset), batch_size=CFG.batch_size, shuffle=False)

    # Initialize HMM Structure
    ns = juice.structures.HMM(
        seq_length=seq_length,
        num_latents=CFG.latents,
        homogeneous=False,
        num_emits=2,
    )
    pc = juice.compile(ns).to(CFG.device)

    print(f"\nStarting training for {tag} ({seq_length} features)...")
    
    with open(log_filename, "w") as log_file:
        for epoch in range(1, CFG.num_epochs + 1):
            t0 = time.time()
            pc.init_param_flows(flows_memory=0.0)

            # Training loop
            train_ll = 0.0
            for batch in train_loader:
                x = batch[0].to(CFG.device)
                lls = pc(x)
                lls.mean().backward()
                train_ll += lls.mean().item()
            
            pc.mini_batch_em(step_size=1.0, pseudocount=CFG.ps)
            train_ll /= len(train_loader)

            # Validation loop
            t1 = time.time()
            valid_ll = 0.0
            with torch.no_grad():
                for batch in valid_loader:
                    x = batch[0].to(CFG.device)
                    valid_ll += pc(x).mean().item()
            valid_ll /= len(valid_loader)
            t2 = time.time()

            stats = (f"[Epoch {epoch}/{CFG.num_epochs}] Train LL: {train_ll:.4f}; "
                     f"Val LL: {valid_ll:.4f} | Time: {t2-t0:.2f}s")
            print(stats)
            log_file.write(stats + "\n")
            log_file.flush()

            if epoch % 1000 == 0 or epoch == CFG.num_epochs:
                save_path = f"pc_{CFG.output_prefix}_{tag}_epoch{epoch}.jpc"
                juice.save(save_path, pc)

    # Cleanup to prevent memory fragmentation between chunks
    del pc, ns, train_loader, valid_loader
    torch.cuda.empty_cache()

def main():
    train_data, valid_data = load_data()
    
    if CFG.num_chunks > 1:
        assert train_data.shape[1] % CFG.num_chunks == 0, "Features must be divisible by num_chunks"
        train_chunks = torch.chunk(train_data, CFG.num_chunks, dim=1)
        valid_chunks = torch.chunk(valid_data, CFG.num_chunks, dim=1)
        
        for i in range(CFG.num_chunks):
            train_model(train_chunks[i], valid_chunks[i], i + 1)
    else:
        # Train on the whole dataset as one block
        train_model(train_data, valid_data, 0)

if __name__ == "__main__":
    main()