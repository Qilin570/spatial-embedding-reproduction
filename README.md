# Spatial Embedding for Selectivity Estimation

This project reproduces the spatial embedding approach for estimating range query selectivity, self-join selectivity, and MBR test counts on spatial data. The original paper proposes a two-stage learning pipeline, and this repo provides a clean, modular reimplementation to validate the key results.

## How It Works

The pipeline has two stages:

1. **M1 (Autoencoder)** -- compresses 128x128x6 spatial histograms into compact embeddings. We test 8 autoencoder configurations across two architectures (Stacked Dense / CNN), different latent dimensions (48 to 3072), and two training regimes (synthetic only vs. synthetic+real).

2. **M2 (Prediction Model)** -- takes those embeddings as input and predicts query selectivity or MBR test counts. We evaluate DNN and CNN architectures at five hyperparameter scales (dH1--dH5 for DNN, cH1--cH5 for CNN) on range query, self-join, and binary join tasks.

## Project Layout

```
my-spatial-embedding/
├── configs.py                  # All AE/M2 configs, normalization constants
├── run_all.py                  # Main entry point
│
├── data/
│   ├── download_data.py        # Download from Mendeley
│   ├── prepare_data.py         # Extract zips, create standardized symlinks
│   ├── histograms.py           # Histogram generation and loading
│   ├── input_gen.py            # M2 input embedding generation (RQ/JN)
│   └── normalization.py        # log(1+cx) + min-max normalization
│
├── models/
│   ├── autoencoders.py         # AE architectures (CNN, Stacked, Global)
│   ├── m2_rq.py                # M2 models for range query
│   └── m2_jn.py                # M2 models for join tasks
│
├── training/
│   ├── train_ae.py             # AE training loop
│   └── train_m2.py             # M2 training loop
│
├── evaluation/
│   └── metrics.py              # WMAPE, MAPE, RMA
│
├── experiments/
│   ├── table3.py               # CNN AE on synthetic data
│   ├── table4.py               # AE on synthetic + real data
│   ├── table5.py               # RQ selectivity (best configs)
│   ├── table5_cv.py            # RQ selectivity (5-fold CV)
│   ├── table6.py               # Self-join selectivity
│   ├── table7.py               # Self-join MBR tests
│   ├── table8.py               # Binary join selectivity
│   ├── table9.py               # Binary join MBR tests
│   ├── table14.py              # RQ full DNN hyperparameter scan
│   ├── table18.py              # BJ full scan (DNN + CNN)
│   └── tsne_viz.py             # t-SNE visualization
│
└── results/                    # Output CSVs and figures
```

## Getting Started

**Install dependencies:**

```bash
pip install tensorflow numpy pandas scikit-learn scipy matplotlib
```

**Prepare data** -- either link from an existing copy of the original repo, or download directly from Mendeley:

```bash
# Option A: link from local spatial-embedding repo
python run_all.py --download --spatial-emb-dir ../spatial-embedding

# Option B: download from Mendeley
python run_all.py --download
```

Then set up the standardized file names:

```bash
python -m data.prepare_data
```

**Run experiments:**

```bash
python run_all.py --tables all          # everything
python run_all.py --tables 3 4 5        # specific tables
python run_all.py --tables 99           # t-SNE visualization only
```

Results go to `results/` as CSV files and PNG figures.

## Experiment Overview

| Table | Stage | What it does |
|-------|-------|--------------|
| 3     | M1    | CNN AE reconstruction (synthetic data) |
| 4     | M1    | AE reconstruction (synthetic + real) |
| 14    | M2    | RQ selectivity -- full DNN scan |
| 5     | M2    | RQ selectivity -- best configs |
| 5 (CV)| M2    | RQ selectivity -- 5-fold cross-validation |
| 6     | M2    | Self-join selectivity |
| 7     | M2    | Self-join MBR tests |
| 18    | M2    | Binary join selectivity -- full scan |
| 8     | M2    | Binary join selectivity -- best configs |
| 9     | M2    | Binary join MBR tests |
| 99    | Viz   | t-SNE embedding quality analysis |

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

## Data

The dataset is hosted on Mendeley Data: [DOI 10.17632/zp9fh6scw9.2](https://data.mendeley.com/datasets/zp9fh6scw9/2)
