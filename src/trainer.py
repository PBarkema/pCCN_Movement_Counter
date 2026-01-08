import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
import copy # Needed to make a copy of the weights

def gaussian_nll_loss(mean, log_var, target):
    var = torch.exp(log_var)
    # Loss = log(var) + Error^2 / var
    return (0.5 * (log_var + (target - mean)**2 / var)).mean()

def train_hybrid_system(train_loader, val_loader, model, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Init Model (324 Features!)
    model = model.to(device)

    # 3. Optimize
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    # Checkpointing logic
    best_nll = float('inf')
    best_weights = copy.deepcopy(model.state_dict())

    print("Starting Hybrid Training...")

    for epoch in range(150):
        # Train
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            mean, log_var = model(x)
            loss = gaussian_nll_loss(mean, log_var, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                mean, log_var = model(x)
                val_loss += gaussian_nll_loss(mean, log_var, y).item()

        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_nll:
            best_nll = avg_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            print(f"Epoch {epoch+1}: New Best NLL {best_nll:.4f}")
        elif (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train NLL {train_loss/len(train_loader):.4f} | Val NLL {avg_val_loss:.4f}")

    print("Loading best model...")
    model.load_state_dict(best_weights)
    return model, val_loader
