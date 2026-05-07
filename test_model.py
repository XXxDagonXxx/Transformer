import torch
import numpy as np
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from configs.config import config
from models.transformer import TransformerEncoder
from models.losses import RPPGLoss

def test_model():
    print("Testing model...")
    
    # Create dummy data
    batch_size = 4
    seq_len = config.sequence_length
    
    # Random RGB input
    dummy_input = torch.randn(batch_size, seq_len, 3)
    
    # Initialize model
    model = TransformerEncoder(
        input_dim=3,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_seq_length=config.max_seq_length
    ).to(config.device)
    
    print(f"Model created successfully")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Move input to device
    dummy_input = dummy_input.to(config.device)
    
    # Forward pass
    try:
        pred_signal, pred_bpm = model(dummy_input)
        print(f"Forward pass successful!")
        print(f"Pred signal shape: {pred_signal.shape}")
        print(f"Pred BPM shape: {pred_bpm.shape}")
        
        # Test loss
        criterion = RPPGLoss()
        target_signal = torch.randn_like(pred_signal)
        target_bpm = torch.randn(batch_size, 1).to(config.device)
        
        loss, loss_dict = criterion(pred_signal, pred_bpm, target_signal, target_bpm, config.fps)
        print(f"Loss computation successful: {loss.item():.4f}")
        print(f"Loss dict: {loss_dict}")
        
        # Test backward pass
        loss.backward()
        print("Backward pass successful!")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset():
    print("\nTesting dataset...")
    
    try:
        from data.dataset import UBFCDataset
        
        if not os.path.exists(config.data_root):
            print(f"Data root does not exist: {config.data_root}")
            print("Please update config.data_root in configs/config.py")
            return False
        
        dataset = UBFCDataset(
            data_root=config.data_root,
            sequence_length=config.sequence_length,
            fps=config.fps,
            is_train=True
        )
        
        print(f"Dataset created with {len(dataset)} samples")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample shapes:")
            print(f"  RGB signal: {sample[0].shape}")
            print(f"  GT signal: {sample[1].shape}")
            print(f"  GT BPM: {sample[2].shape}")
            return True
        else:
            print("No samples found in dataset")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*50)
    print("Testing rPPG Transformer Pipeline")
    print("="*50)
    
    model_ok = test_model()
    
    if model_ok:
        print("\n✓ Model test passed!")
    else:
        print("\n✗ Model test failed!")
    
    dataset_ok = test_dataset()
    
    if dataset_ok:
        print("\n✓ Dataset test passed!")
    else:
        print("\n✗ Dataset test failed!")
    
    if model_ok and dataset_ok:
        print("\n" + "="*50)
        print("All tests passed! Ready to train.")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("Some tests failed. Please fix the issues above.")
        print("="*50)
