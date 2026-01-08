import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from collections import Counter
import copy

# ==========================================
# 1. THE FEATURE INJECTION LOGIC
# ==========================================
def generate_smart_signal(video_tensor):
    """
    Takes raw video (300, 323).
    Returns (300, 1) "Smart Signal" (PCA + Smooth + Orient).
    """
    # Convert to Numpy for Scikit-Learn/Scipy
    matrix = video_tensor.numpy()

    # 1. PCA (Extract Dominant Motion)
    # Standardize first to prevent one feature dominating
    matrix_std = (matrix - matrix.mean(axis=0)) / (matrix.std(axis=0) + 1e-6)

    # Fit PCA on this specific video
    pca = PCA(n_components=1)
    signal = pca.fit_transform(matrix_std).flatten()

    # 2. SMOOTHING (Savgol Filter)
    try:
        signal = savgol_filter(signal, window_length=15, polyorder=3)
    except ValueError:
        pass # Fallback if video is too short (rare)

    # 3. ORIENTATION NORMALIZATION
    # Ensure the "Pushup" is always a POSITIVE deviation
    baseline = np.median(signal[:5]) # Resting state
    if abs(np.min(signal) - baseline) > abs(np.max(signal) - baseline):
        signal = -signal # Flip if the deep part is deeper than the high part

    # 4. SCALE NORMALIZATION
    # Squash to roughly -1 to 1 range so the CNN handles it easily
    signal = (signal - signal.mean()) / (signal.std() + 1e-6)

    return torch.tensor(signal, dtype=torch.float32).unsqueeze(1) # (300, 1)

# ==========================================
# 2. HYBRID DATASET
# ==========================================
class HybridPoseDataset(Dataset):
    def __init__(self, data_tensor, label_tensor, augment=False):
        self.data = data_tensor
        self.labels = label_tensor
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Shape: (300, 323)
        x = self.data[idx].clone()
        y = self.labels[idx].clone()

        # --- INJECTION STEP ---
        # Generate the "Cheat Sheet" signal
        # We do this BEFORE augmentation so the signal is clean
        smart_signal = generate_smart_signal(x) # (300, 1)

        # Concatenate: (300, 323) + (300, 1) -> (300, 324)
        x_hybrid = torch.cat([x, smart_signal], dim=1)

        # Augmentation (Noise)
        if self.augment:
            noise = torch.randn_like(x_hybrid) * 0.005
            x_hybrid += noise

        return x_hybrid.float(), y.float()

def create_hybrid_data_loaders(data_path, batch_size=16):
    print(f"Loading {data_path}...")
    data_dict = torch.load(data_path)
    data = data_dict['data']
    labels = data_dict['labels']

    total_count = len(labels)
    # Deterministic Split
    indices = torch.randperm(total_count, generator=torch.Generator().manual_seed(42))
    split = int(total_count * 0.2)
    val_indices = indices[:split]
    train_indices = indices[split:]

    # Class Balancing
    train_labels = labels[train_indices]
    counts = Counter(train_labels.long().tolist())
    weights = {cls: 1.0/(count + 1e-6) for cls, count in counts.items()}
    sample_weights = [weights.get(int(l), 0) for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(train_indices), replacement=True)

    # Create Datasets using the HYBRID class
    train_ds = HybridPoseDataset(data[train_indices], labels[train_indices], augment=True)
    val_ds = HybridPoseDataset(data[val_indices], labels[val_indices], augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_loader, val_loader