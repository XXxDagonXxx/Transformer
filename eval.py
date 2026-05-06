import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import sys
from scipy.stats import pearsonr
from scipy.fft import fft, fftfreq

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import config
from data.dataset import UBFCDataset
from models.transformer import TransformerEncoder

def compute_metrics(pred_bpm, gt_bpm):
    """Compute MAE, RMSE, and Pearson correlation"""
    mae = np.mean(np.abs(pred_bpm - gt_bpm))
    rmse = np.sqrt(np.mean((pred_bpm - gt_bpm) ** 2))
    
    if len(pred_bpm) > 1:
        pearson, _ = pearsonr(pred_bpm, gt_bpm)
    else:
        pearson = 0.0
    
    return mae, rmse, pearson

def compute_bpm_from_signal(signal, fs=30):
    """Compute BPM from rPPG signal using FFT"""
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1/fs)
    
    # Get positive frequencies within bandpass range
    mask = (xf >= 0.7) & (xf <= 4.0) & (xf > 0)
    freqs = xf[mask]
    magnitudes = np.abs(yf[mask])
    
    if len(magnitudes) > 0:
        peak_freq = freqs[np.argmax(magnitudes)]
        bpm = peak_freq * 60
        return bpm
    return 0.0

def evaluate(model, dataloader, device, fs=30):
    model.eval()
    all_pred_bpm = []
    all_gt_bpm = []
    all_pred_signals = []
    all_gt_signals = []
    
    with torch.no_grad():
        for batch_idx, (rgb_signal, gt_signal, gt_bpm) in enumerate(tqdm(dataloader, desc="Evaluating")):
            rgb_signal = rgb_signal.to(device)
            gt_signal = gt_signal.to(device)
            gt_bpm = gt_bpm.numpy()
            
            # Forward pass
            pred_signal, pred_bpm = model(rgb_signal)
            
            pred_signal = pred_signal.cpu().numpy()
            pred_bpm = pred_bpm.cpu().numpy().flatten()
            
            # Compute BPM from predicted signal using FFT
            for i in range(len(pred_signal)):
                sig = pred_signal[i]
                bpm_from_signal = compute_bpm_from_signal(sig, fs)
                all_pred_bpm.append(bpm_from_signal)
            
            all_gt_bpm.extend(gt_bpm.flatten())
            all_pred_signals.extend(pred_signal)
            all_gt_signals.extend(gt_signal.numpy())
    
    all_pred_bpm = np.array(all_pred_bpm)
    all_gt_bpm = np.array(all_gt_bpm)
    
    # Compute metrics
    mae, rmse, pearson = compute_metrics(all_pred_bpm, all_gt_bpm)
    
    results = {
        'MAE': mae,
        'RMSE': rmse,
        'Pearson': pearson,
        'pred_bpm': all_pred_bpm,
        'gt_bpm': all_gt_bpm
    }
    
    return results

def main():
    # Dataset and DataLoader
    test_dataset = UBFCDataset(
        data_root=config.data_root,
        sequence_length=config.sequence_length,
        fps=config.fps,
        is_train=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Load model
    model = TransformerEncoder(
        input_dim=3,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_seq_length=config.max_seq_length
    ).to(config.device)
    
    checkpoint = torch.load(
        os.path.join(config.save_dir, config.model_name),
        map_location=config.device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Evaluate
    results = evaluate(model, test_loader, config.device, config.fps)
    
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    print(f"MAE: {results['MAE']:.2f} BPM")
    print(f"RMSE: {results['RMSE']:.2f} BPM")
    print(f"Pearson Correlation: {results['Pearson']:.4f}")
    print("="*50)
    
    # Save results
    np.save(os.path.join(config.save_dir, 'eval_results.npy'), results)
    print(f"Results saved to {os.path.join(config.save_dir, 'eval_results.npy')}")

if __name__ == "__main__":
    main()
