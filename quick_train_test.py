"""
Quick test: Train for 2 epochs with 2 samples to verify pipeline works
"""
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import sys
from tqdm import tqdm

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from configs.config import config
from data.dataset import UBFCDataset
from models.transformer import TransformerEncoder
from models.losses import RPPGLoss

# Override config for quick test
config.num_epochs = 2
config.batch_size = 2
config.num_workers = 0

print("="*60)
print("QUICK TRAINING TEST (2 epochs, 2 samples)")
print("="*60)

# Create small dataset
print("\nLoading small dataset...")
full_dataset = UBFCDataset(
    data_root=config.data_root,
    sequence_length=config.sequence_length,
    fps=config.fps,
    is_train=True
)

# Use only first 2 samples
indices = [0, 1] if len(full_dataset) >= 2 else [0]
small_dataset = Subset(full_dataset, indices=indices)

train_loader = DataLoader(
    small_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=0
)

print(f"Dataset size: {len(small_dataset)}")

# Model
print("\nInitializing model...")
model = TransformerEncoder(
    input_dim=3,
    d_model=64,  # Smaller for quick test
    nhead=4,
    num_encoder_layers=2,
    dim_feedforward=128,
    dropout=0.1,
    max_seq_length=config.max_seq_length
).to(config.device)

criterion = RPPGLoss(
    signal_weight=config.signal_loss_weight,
    bpm_weight=config.bpm_loss_weight,
    freq_weight=config.freq_loss_weight
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay
)

# Quick training
print("\nStarting quick training...")
for epoch in range(config.num_epochs):
    print(f"\nEpoch {epoch+1}/{config.num_epochs}")
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (rgb_signal, gt_signal, gt_bpm) in enumerate(tqdm(train_loader, desc="Training")):
        if rgb_signal is None:
            continue
        
        rgb_signal = rgb_signal.to(config.device)
        gt_signal = gt_signal.to(config.device)
        gt_bpm = gt_bpm.to(config.device)
        
        optimizer.zero_grad()
        pred_signal, pred_bpm = model(rgb_signal)
        loss, loss_dict = criterion(pred_signal, pred_bpm, gt_signal, gt_bpm, config.fps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        print(f"  Batch {batch_idx}: loss={loss.item():.4f}")
    
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        print(f"Average loss: {avg_loss:.4f}")

print("\n" + "="*60)
print("QUICK TEST COMPLETE!")
print("="*60)
print("If you see this message, the training pipeline works!")
print("Run 'python train.py' for full training.")
