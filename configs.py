"""Centralized configuration for all experiments.

Constants and hyperparameters extracted from the authors' code:
  spatial-embedding/modelsRQ/gen_py/generate_input_RQ.py (normalization constants, reference space)
  spatial-embedding/modelsSJ/gen_py/generate_input_JN.py (join reference space)
  spatial-embedding/autoEncoders/code_py/run_autoenc.py (AE hyperparameters)
  spatial-embedding/modelsRQ/code_py/run_model_all.py (M2 hyperparameters)
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import math


@dataclass
class AutoencoderConfig:
    name: str
    ae_type: str  # "cnn" or "stacked"
    latent_dim: int
    f1: int
    f2: int
    trained_on: str  # "synthetic" or "synthetic+real"

    @property
    def emb_shape(self) -> Tuple[int, int, int]:
        """Return (x, y, z) reshape dimensions for the embedding."""
        return LATENT_DIM_TO_SHAPE[self.latent_dim]

    @property
    def model_filename(self) -> str:
        """Return the expected model filename for this AE."""
        return AE_MODEL_FILES[self.name]


@dataclass
class M2HyperparamConfig:
    name: str
    m2_type: str  # "dnn" or "cnn"
    filters: List[int]

    @property
    def label(self) -> str:
        return f"{self.name}({','.join(str(f) for f in self.filters)})"


# --- Autoencoder Configurations (Table 2/3/4) ---

AE_CONFIGS: Dict[str, AutoencoderConfig] = {
    # Trained on synthetic data only
    "AE_S1": AutoencoderConfig("AE_S1", "stacked", 384,  1024, 512, "synthetic"),
    "AE_S2": AutoencoderConfig("AE_S2", "stacked", 1536, 1024, 512, "synthetic"),
    "AE_C1": AutoencoderConfig("AE_C1", "cnn",     768,  128,  64,  "synthetic"),
    "AE_C2": AutoencoderConfig("AE_C2", "cnn",     3072, 64,   32,  "synthetic"),
    # Trained on synthetic + real data
    "AE_S3": AutoencoderConfig("AE_S3", "stacked", 48,   16,   32,  "synthetic+real"),
    "AE_S4": AutoencoderConfig("AE_S4", "stacked", 384,  16,   32,  "synthetic+real"),
    "AE_C3": AutoencoderConfig("AE_C3", "cnn",     1536, 128,  64,  "synthetic+real"),
    "AE_C4": AutoencoderConfig("AE_C4", "cnn",     768,  64,   32,  "synthetic+real"),
}

# Model file names mapping
AE_MODEL_FILES = {
    "AE_S1": "autoencoder_DENSE3L_1024-512_emb384_synthetic",
    "AE_S2": "autoencoder_DENSE3L_1024-512_emb1536_synthetic",
    "AE_C1": "autoencoder_CNN_128-64_emb768_synthetic",
    "AE_C2": "autoencoder_CNN_64-32_emb3072_synthetic",
    "AE_S3": "autoencoder_DENSE3L_16-32_emb48_real",
    "AE_S4": "autoencoder_DENSE3L_16-32_emb384_real",
    "AE_C3": "autoencoder_CNN3L_128-64_emb1536_real",
    "AE_C4": "autoencoder_CNN3L_64-32_emb768_real",
}

# Latent dimension -> (x, y, z) embedding shape
LATENT_DIM_TO_SHAPE = {
    3072: (32, 32, 3),
    1536: (32, 16, 3),
    768:  (16, 16, 3),
    384:  (16, 8,  3),
    48:   (4,  4,  3),
}

# --- M2 Hyperparameter Configurations ---

# DNN hyperparams: 5 hidden layers [f1, f2, f3, f4, f5]
M2_DNN_CONFIGS = {
    "dH1": M2HyperparamConfig("dH1", "dnn", [64, 32, 32, 16, 16]),
    "dH2": M2HyperparamConfig("dH2", "dnn", [128, 64, 64, 32, 32]),
    "dH3": M2HyperparamConfig("dH3", "dnn", [256, 128, 128, 64, 64]),
    "dH4": M2HyperparamConfig("dH4", "dnn", [512, 256, 256, 128, 128]),
    "dH5": M2HyperparamConfig("dH5", "dnn", [1024, 512, 512, 256, 256]),
}

# CNN hyperparams: 4 values [f1, f2, f3, f4]
M2_CNN_CONFIGS = {
    "cH1": M2HyperparamConfig("cH1", "cnn", [64, 32, 32, 16]),
    "cH2": M2HyperparamConfig("cH2", "cnn", [128, 64, 64, 32]),
    "cH3": M2HyperparamConfig("cH3", "cnn", [256, 128, 128, 64]),
    "cH4": M2HyperparamConfig("cH4", "cnn", [512, 256, 256, 128]),
    "cH5": M2HyperparamConfig("cH5", "cnn", [1024, 512, 512, 256]),
}

# --- Histogram Constants ---

DIM_H_X = 128
DIM_H_Y = 128
DIM_H_Z = 6
DIM_HG_Z = 1

# --- Normalization Constants ---
# Extracted from the authors' code: generate_input_RQ.py

NORM_MIN = [0., 0., 0., 0., 0., 0.]
# Synthetic only
NORM_MAX_SYNTHETIC = [8.77805800e+06, 3.05404802e+09, 1.53571255e+08,
                      3.03019291e-02, 1.91233400e-01, 2.20753674e-01]
# Synthetic + real
NORM_MAX_REAL = [8.77805800e+06, 3.05404802e+09, 1.53571255e+08,
                 1.00000000e+00, 1.00000000e+00, 1.00000000e+00]

NORM_MIN_G = 0.0
NORM_MAX_G = 8708693.144550692

# --- Reference Space ---
# Extracted from the authors' code: generate_input_RQ.py, generate_input_JN.py

# Range Query reference space
RQ_X_MIN_REF = 0
RQ_X_MAX_REF = 10
RQ_Y_MIN_REF = 0
RQ_Y_MAX_REF = 10

# Join reference space
JN_X_MIN_REF = 0
JN_X_MAX_REF = 20
JN_Y_MIN_REF = 0
JN_Y_MAX_REF = 20

# Global histogram reference space
GLOBAL_X_MIN = 0
GLOBAL_Y_MIN = 0
GLOBAL_X_MAX = 10
GLOBAL_Y_MAX = 10

# --- Training Constants ---
# Extracted from the authors' code: run_autoenc.py, run_model_all.py

AE_BATCH_SIZE = 16
AE_EPOCHS = 50
AE_VALIDATION_SPLIT = 0.2

M2_BATCH_SIZE = 8
M2_EPOCHS = 80
M2_PATIENCE = 6
M2_VALIDATION_SPLIT = 0.2

# --- Global Encoder ---

GLOBAL_ENCODER_FILE = "model_2048_CNNDense_newDatasets_SMLG_global_new"
GLOBAL_ENCODER_LATENT_DIM = 2048
GLOBAL_EMB_SHAPE = (32, 32, 2)


def get_norm_max(trained_on: str) -> list:
    """Return NORM_MAX based on training data type."""
    if trained_on == "synthetic+real":
        return NORM_MAX_REAL
    return NORM_MAX_SYNTHETIC
