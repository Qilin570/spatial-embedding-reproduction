# Spatial Embedding for Selectivity Estimation

A clean reimplementation of the spatial embedding approach for estimating range query selectivity, self-join selectivity, and MBR test counts on spatial data.

## Architecture

The system uses a two-stage pipeline:

**M1 (Autoencoder)** compresses 128x128x6 spatial histograms into low-dimensional embeddings. Eight configurations cover two architectures (Stacked / CNN), multiple latent dimensions (48--3072), and two training regimes (synthetic only / synthetic+real).

**M2 (Prediction Model)** takes the embeddings and predicts query results. Two architectures (DNN / CNN) with five hyperparameter scales each (dH1--dH5, cH1--cH5) are evaluated across range query, self-join, and binary join tasks.

## Project Structure

```
my-spatial-embedding/
├── configs.py                  # All AE/M2 configs, normalization constants
├── run_all.py                  # Entry point for running experiments
│
├── data/
│   ├── download_data.py        # Download data from Mendeley
│   ├── prepare_data.py         # Extract zips, create standardized symlinks
│   ├── histograms.py           # Histogram generation and loading
│   ├── input_gen.py            # M2 input embedding generation (RQ/JN)
│   ├── normalization.py        # Min-max + log normalization utilities
│   └── downloaded_data/        # Data files (not tracked in git)
│
├── models/
│   ├── autoencoders.py         # AE architectures (CNN, Stacked, Global)
│   ├── m2_rq.py                # M2 models for range query
│   └── m2_jn.py                # M2 models for self-join / binary join
│
├── training/
│   ├── train_ae.py             # AE training (MSE loss, Adam, 50 epochs)
│   └── train_m2.py             # M2 training (MAE loss, EarlyStopping, 80 epochs)
│
├── evaluation/
│   └── metrics.py              # WMAPE, MAPE, RMA metrics
│
├── experiments/
│   ├── table3.py               # CNN AE reconstruction (synthetic)
│   ├── table4.py               # AE reconstruction (synthetic + real)
│   ├── table5.py               # RQ selectivity best results
│   ├── table5_cv.py            # RQ selectivity 5-fold cross-validation
│   ├── table6.py               # Self-join selectivity best results
│   ├── table7.py               # Self-join MBR tests best results
│   ├── table8.py               # Binary join selectivity best results
│   ├── table9.py               # Binary join MBR tests best results
│   ├── table14.py              # RQ selectivity full DNN scan
│   ├── table18.py              # BJ selectivity full scan (DNN + CNN)
│   └── tsne_viz.py             # t-SNE embedding quality visualization
│
└── results/                    # Output CSVs and figures
```

## Quick Start

### 1. Install Dependencies

```bash
pip install tensorflow numpy pandas scikit-learn scipy matplotlib
```

### 2. Prepare Data

Option A -- link from the original `spatial-embedding` repo:

```bash
python run_all.py --download --spatial-emb-dir ../spatial-embedding
```

Option B -- download from Mendeley:

```bash
python run_all.py --download
```

Then create standardized symlinks:

```bash
python -m data.prepare_data
```

### 3. Run Experiments

```bash
# Run all experiments
python run_all.py --tables all

# Run specific tables
python run_all.py --tables 3 4 5

# Run t-SNE visualization only
python run_all.py --tables 99
```

Results are saved to `results/` as CSV files and PNG figures.

## Experiments

| Table | Task | Description |
|-------|------|-------------|
| 3     | M1   | CNN AE reconstruction quality (synthetic data) |
| 4     | M1   | AE reconstruction quality (synthetic + real data) |
| 14    | M2   | RQ selectivity -- full DNN hyperparameter scan |
| 5     | M2   | RQ selectivity -- best configurations |
| 51  | M2   | RQ selectivity -- 5-fold cross-validation |
| 6     | M2   | Self-join selectivity -- best configurations |
| 7     | M2   | Self-join MBR tests -- best configurations |
| 18    | M2   | Binary join selectivity -- full scan |
| 8     | M2   | Binary join selectivity -- best configurations |
| 9     | M2   | Binary join MBR tests -- best configurations |
| 99    | Viz  | t-SNE embedding quality analysis |

## Autoencoder Configurations

| Name  | Type    | Latent Dim | Training Data    |
|-------|---------|-----------|------------------|
| AE_S1 | Stacked | 384       | Synthetic        |
| AE_S2 | Stacked | 1536      | Synthetic        |
| AE_C1 | CNN     | 768       | Synthetic        |
| AE_C2 | CNN     | 3072      | Synthetic        |
| AE_S3 | Stacked | 48        | Synthetic + Real |
| AE_S4 | Stacked | 384       | Synthetic + Real |
| AE_C3 | CNN     | 1536      | Synthetic + Real |
| AE_C4 | CNN     | 768       | Synthetic + Real |

## Data Source

Dataset: [Mendeley Data DOI 10.17632/zp9fh6scw9.2](https://data.mendeley.com/datasets/zp9fh6scw9/2)
