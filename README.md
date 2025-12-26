# DerivOmniModel - Multi-Expert Trading System

A modular deep learning system for binary options trading on Deriv.com.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DerivOmniModel                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   Temporal   │  │   Spatial    │  │    Volatility        │   │
│  │    Expert    │  │    Expert    │  │      Expert          │   │
│  │  (BiLSTM)    │  │   (CNN)      │  │   (Autoencoder)      │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│         │                 │                    │                │
│         └─────────────────┼────────────────────┘                │
│                           │                                      │
│                    ┌──────┴──────┐                              │
│                    │   Fusion    │                              │
│                    └──────┬──────┘                              │
│         ┌─────────────────┼─────────────────┐                   │
│  ┌──────┴─────┐    ┌──────┴─────┐    ┌──────┴─────┐            │
│  │ Rise/Fall  │    │   Touch    │    │   Range    │            │
│  │    Head    │    │    Head    │    │    Head    │            │
│  └────────────┘    └────────────┘    └────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

Copy `.env.example` to `.env` and set your Deriv API token:

```bash
DERIV_API_TOKEN=your_token_here
TRADING__SYMBOL=R_100
```

### 3. Download Historical Data

```bash
python scripts/download_data.py --months 12
```

### 4. Train (Local or Colab)

**Local:**
```bash
python scripts/train.py --epochs 50 --batch-size 128
```

**Colab:**
1. Upload project folder to Google Drive
2. Open `notebooks/train_colab.ipynb` in Colab
3. Run all cells

### 5. Run Live Trading

```bash
python scripts/live.py --checkpoint best_model
```

## Project Structure

```
xtitan/
├── config/             # Configuration (settings, constants)
├── data/               # Data loading, ingestion, and feature engineering
├── models/             # Neural network modules (DerivOmniModel)
├── training/           # Training loop and metrics
├── execution/          # Trading decision logic and safety
├── utils/              # Utilities (device, seed)
├── scripts/            # CLI entry points
├── notebooks/          # Jupyter/Colab notebooks
├── checkpoints/        # Saved model weights
└── data_cache/         # Historical data and SQLite Stores (Shadow/Safety)
```

## Key Commands

| Command | Description |
|---------|-------------|
| `python scripts/download_data.py --months 12` | Download 12 months of data |
| `python scripts/train.py --epochs 50` | Train model locally |
| `python scripts/live.py --test` | Test Deriv API connection |
| `python scripts/live.py --checkpoint best_model` | Run live trading |
| `python scripts/live.py --compound --x-amount 2.0` | Run with compounding strategy |
| `python scripts/generate_shadow_report.py` | Generate HTML analysis of shadow trades |

## Model Details

- **Temporal Expert**: BiLSTM with attention for directional prediction.
- **Spatial Expert**: Pyramidal CNN for price geometry analysis.
- **Volatility Expert**: Autoencoder for regime detection and anomaly vetoing.
- **CalibrationMonitor**: Real-time tracking of model probability accuracy and graceful degradation.
- **Fusion Layer**: Combines expert embeddings into actionable signals.
- **Contract Heads**: Separate heads for Rise/Fall, Touch, Range contracts

## Training

The model uses:
- **Multi-task learning** with weighted loss for each contract type
- **Early stopping** based on validation loss
- **Cosine annealing** learning rate schedule
- **TensorBoard** logging for monitoring

Labels are generated automatically from historical data:
- **Rise/Fall**: Future close > current close
- **Touch**: Price moved > 0.5% from current
- **Range**: Price stayed within 0.3% band

## License

MIT
