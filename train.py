import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import config
from data.dataset import UBFCDataset
from models.transformer import TransformerEncoder
from models.losses import RPPGLoss

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    loss_dicts = []
    
    for batch_idx, (rgb_signal, gt_signal, gt_bpm) in enumerate(tqdm(dataloader, desc="Training")):
        rgb_signal = rgb_signal.to(device)
        gt_signal = gt_signal.to(device)
        gt_bpm = gt_bpm.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_signal, pred_bpm = model(rgb_signal)
        
        # Compute loss
        loss, loss_dict = criterion(pred_signal, pred_bpm, gt_signal, gt_bpm)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        loss_dicts.append(loss_dict)
    
    avg_loss = total_loss / len(dataloader)
    avg_loss_dict = {
        k: np.mean([d[k] for d in loss_dicts]) for k in loss_dicts[0].keys()
    }
    
    return avg_loss, avg_loss_dict

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    loss_dicts = []
    
    with torch.no_grad():
        for batch_idx, (rgb_signal, gt_signal, gt_bpm) in enumerate(tqdm(dataloader, desc="Validation")):
            rgb_signal = rgb_signal.to(device)
            gt_signal = gt_signal.to(device)
            gt_bpm = gt_bpm.to(device)
            
            # Forward pass
            pred_signal, pred_bpm = model(rgb_signal)
            
            # Compute loss
            loss, loss_dict = criterion(pred_signal, pred_bpm, gt_signal, gt_bpm)
            
            total_loss += loss.item()
            loss_dicts.append(loss_dict)
    
    avg_loss = total_loss / len(dataloader)
    avg_loss_dict = {
        k: np.mean([d[k] for d in loss_dicts]) for k in loss_dicts[0].keys()
    }
    
    return avg_loss, avg_loss_dict

def main():
    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)
    
    # Dataset and DataLoader
    train_dataset = UBFCDataset(
        data_root=config.data_root,
        sequence_length=config.sequence_length,
        fps=config.fps,
        is_train=True
    )
    
    val_dataset = UBFCDataset(
        data_root=config.data_root,
        sequence_length=config.sequence_length,
        fps=config.fps,
        is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Model
    model = TransformerEncoder(
        input_dim=3,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_seq_length=config.max_seq_length
    ).to(config.device)
    
    # Loss and optimizer
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
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Train
        train_loss, train_loss_dict = train_epoch(
            model, train_loader, criterion, optimizer, config.device
        )
        
        # Validate
        val_loss, val_loss_dict = validate(
            model, val_loader, criterion, config.device
        )
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Losses: {train_loss_dict}")
        print(f"Val Losses: {val_loss_dict}")
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(config.save_dir, config.model_name))
            print(f"Saved best model with val_loss: {val_loss:.4f}")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()
