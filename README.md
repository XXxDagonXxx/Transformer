# Transformer-based rPPG Pipeline

A complete PyTorch implementation of a Transformer-based remote Photoplethysmography (rPPG) pipeline for heart rate estimation from facial videos.

## Features

- **Dataset**: UBFC-rPPG dataset support
- **Face Detection**: MediaPipe-based face detection
- **ROI Extraction**: Forehead and cheeks regions
- **Signal Processing**: RGB temporal signals with bandpass filtering (0.7-4 Hz)
- **Model**: Transformer encoder with positional encoding and multi-head attention
- **Dual Output**: rPPG signal (sequence) + BPM (scalar)
- **Loss Function**: Combined MSE (signal) + MAE (BPM) + FFT-based frequency loss
- **Evaluation**: MAE, RMSE, Pearson correlation

## Project Structure

```
Transformer/
├── configs/
│   ├── __init__.py
│   └── config.py           # Configuration parameters
├── data/
│   ├── __init__.py
│   └── dataset.py          # UBFC-rPPG dataset loader
├── models/
│   ├── __init__.py
│   ├── transformer.py      # Transformer model
│   └── losses.py           # Loss functions
├── utils/
│   ├── __init__.py
│   ├── face_detection.py   # MediaPipe face detection
│   ├── roi_extraction.py   # ROI extraction
│   └── signal_processing.py # Signal processing utilities
├── train.py                # Training script
├── eval.py                 # Evaluation script
├── inference.py            # Inference script
├── requirements.txt        # Dependencies
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Configure Dataset Path

Edit `configs/config.py` and set the correct path to your UBFC-rPPG dataset:

```python
data_root: str = "/path/to/UBFC-rPPG"
```

### 2. Train the Model

```bash
python train.py
```

This will:
- Load the UBFC-rPPG dataset
- Train the Transformer model
- Save the best model to `checkpoints/transformer_rppg.pth`

### 3. Evaluate the Model

```bash
python eval.py
```

This will compute MAE, RMSE, and Pearson correlation on the test set.

### 4. Run Inference

```bash
python inference.py
```

You can either:
- Provide a video path for rPPG prediction
- Use webcam for real-time prediction (press Enter without input)

## Model Architecture

- **Input**: RGB temporal signal of shape (T, 3)
- **Transformer Encoder**:
  - Input projection: 3 → d_model
  - Positional encoding
  - 6 Transformer encoder layers (8 heads, d_model=128)
  - Dropout: 0.1
- **Output Heads**:
  - rPPG signal: (T,) - sequence regression
  - BPM: (1,) - scalar regression

## Configuration

Key parameters in `configs/config.py`:

- `sequence_length`: 300 (T in (T, 3))
- `fps`: 30
- `d_model`: 128
- `nhead`: 8
- `num_encoder_layers`: 6
- `learning_rate`: 1e-4
- `batch_size`: 16
- `num_epochs`: 100

## Loss Function

Combined loss with three components:

1. **Signal Loss**: MSE between predicted and ground truth rPPG signals
2. **BPM Loss**: MAE between predicted and ground truth BPM
3. **Frequency Loss**: MSE in frequency domain using FFT

Weights:
- signal_weight: 1.0
- bpm_weight: 0.5
- freq_weight: 0.3

## Evaluation Metrics

- **MAE**: Mean Absolute Error (BPM)
- **RMSE**: Root Mean Square Error (BPM)
- **Pearson**: Pearson correlation coefficient

## GPU Support

The pipeline automatically detects and uses GPU if available:

```python
device: str = "cuda" if torch.cuda.is_available() else "cpu"
```

## References

- UBFC-rPPG Dataset: https://sites.google.com/view/ybenezeth/ubfc-rppg
- Transformer: Attention Is All You Need (Vaswani et al., 2017)
- MediaPipe: https://developers.google.com/mediapipe
