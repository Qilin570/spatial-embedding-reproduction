"""M1 Autoencoder training.

Training and evaluation functions adapted from the authors' code:
  spatial-embedding/autoEncoders/code_py/run_autoenc.py
"""
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import losses
from sklearn.model_selection import train_test_split

from models.autoencoders import create_autoencoder, create_global_autoencoder
from data.normalization import nor_g_ab, nor_a_ab, denorm_g_ab
import configs as cfg


# Adapted from the authors' code: run_autoenc.py - nor_and_train
def train_autoencoder(ae_config, hist_local, hist_global=None,
                      batch_size=None, epochs=None, val_split=None,
                      norm_min=None, norm_max=None):
    """Train an autoencoder on histogram data.

    Follows the original code's normalization approach: compute min/max from
    the data itself using nor_g_ab(data, 1, -1, -1), or use pre-computed
    norm_min/norm_max if provided.

    Args:
        ae_config: AutoencoderConfig
        hist_local: local histograms array (N, 128, 128, 6)
        hist_global: global histograms array (N, 128, 128) or None
        batch_size: training batch size (default: AE_BATCH_SIZE)
        epochs: number of epochs (default: AE_EPOCHS)
        val_split: validation split ratio (default: AE_VALIDATION_SPLIT)
        norm_min: pre-computed normalization min (original scale, per-feature).
                  If None, computed from hist_local.
        norm_max: pre-computed normalization max (original scale, per-feature).
                  If None, computed from hist_local.
    Returns:
        model: trained autoencoder
        history: training history
        train_time: training time in seconds
        norm_min: normalization min values used (original scale)
        norm_max: normalization max values used (original scale)
    """
    if batch_size is None:
        batch_size = cfg.AE_BATCH_SIZE
    if epochs is None:
        epochs = cfg.AE_EPOCHS
    if val_split is None:
        val_split = cfg.AE_VALIDATION_SPLIT

    # Normalize local histograms
    # Adapted from the authors' code: nor_g_ab(a_tot, 1, -1, -1)
    # Uses data-derived min/max when norm_min/norm_max not provided
    print(f"Normalizing {hist_local.shape[0]} local histograms...")
    if norm_min is None or norm_max is None:
        hist_norm, norm_min, norm_max = nor_g_ab(hist_local.copy(), 1, -1, -1)
    else:
        hist_norm, _, _ = nor_g_ab(hist_local.copy(), 1, norm_min, norm_max)

    # Create model
    model = create_autoencoder(ae_config)

    # Compile
    model.compile(optimizer='adam', loss=losses.MeanSquaredError())

    # Split
    X_train, X_val = train_test_split(hist_norm, test_size=val_split, random_state=42)

    # Train
    print(f"Training {ae_config.name} ({ae_config.ae_type}, LD={ae_config.latent_dim})...")
    print(f"  f1={ae_config.f1}, f2={ae_config.f2}")
    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
    print(f"  Batch: {batch_size}, Epochs: {epochs}")

    t0 = time.time()
    history = model.fit(
        X_train, X_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        validation_data=(X_val, X_val),
    )
    train_time = time.time() - t0

    print(f"  Training time: {train_time:.1f}s")
    print(f"  Final loss: {history.history['loss'][-1]:.6f}")
    print(f"  Final val_loss: {history.history['val_loss'][-1]:.6f}")

    return model, history, train_time, norm_min, norm_max


# Adapted from the authors' code: run_autoenc.py - nor_and_train (global AE variant)
def train_global_autoencoder(hist_global, latent_dim=2048,
                             batch_size=None, epochs=None, val_split=None):
    """Train a global histogram autoencoder.

    Args:
        hist_global: global histograms (N, 128, 128)
        latent_dim: embedding dimension
    Returns:
        model, history, train_time
    """
    if batch_size is None:
        batch_size = cfg.AE_BATCH_SIZE
    if epochs is None:
        epochs = cfg.AE_EPOCHS
    if val_split is None:
        val_split = cfg.AE_VALIDATION_SPLIT

    # Normalize
    hist_norm, _, _ = nor_g_ab(hist_global.copy(), 1, cfg.NORM_MIN_G, cfg.NORM_MAX_G)

    # Reshape for CNN: add channel dimension
    hist_norm = hist_norm.reshape((-1, 128, 128, 1))

    model = create_global_autoencoder(latent_dim)
    model.compile(optimizer='adam', loss=losses.MeanSquaredError())

    X_train, X_val = train_test_split(hist_norm, test_size=val_split, random_state=42)

    print(f"Training global AE (LD={latent_dim})...")
    t0 = time.time()
    history = model.fit(
        X_train, X_train,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        validation_data=(X_val, X_val),
    )
    train_time = time.time() - t0

    return model, history, train_time


# Adapted from the authors' code: run_autoenc.py - wmape + denormalization logic
def evaluate_autoencoder(model, hist_data, norm_min, norm_max,
                         use_encoder_decoder=False):
    """Evaluate autoencoder reconstruction quality.

    Computes WMAPE on denormalized (original scale) data, using per-feature
    equal-weight averaging as in the original paper.

    Flow: original → normalize → encode → decode → denormalize → compare with original

    Args:
        model: trained autoencoder
        hist_data: original histograms (unnormalized)
        norm_min: normalization min values (original scale, per-feature)
        norm_max: normalization max values (original scale, per-feature)
        use_encoder_decoder: if True, call encoder/decoder separately instead of
            model.predict(). Required for Dense (stacked) autoencoders loaded
            from TF2 SavedModel, where model.predict() produces incorrect output
            due to traced call() graph issues.
    Returns:
        wmape: arithmetic mean of per-feature WMAPEs
        wmape_per_feature: per-feature WMAPE
    """
    # Normalize
    hist_norm, _, _ = nor_g_ab(hist_data.copy(), 1, norm_min, norm_max)

    # Reconstruct.
    # Adapted from the authors' code: run_autoenc.py - nor_and_train
    #   enc_a_test = autoenc.encoder(a_test).numpy()
    #   dec_a_test = autoenc.decoder(enc_a_test).numpy()
    if use_encoder_decoder:
        # Use separate encoder/decoder calls. This is required for Dense AEs
        # loaded from SavedModel, where the traced call() graph is broken.
        encoded = model.encoder(hist_norm).numpy()
        reconstructed_norm = model.decoder(encoded).numpy()
    else:
        reconstructed_norm = model.predict(hist_norm, verbose=0)

    # Denormalize reconstructed data back to original scale
    reconstructed = denorm_g_ab(reconstructed_norm, 1, norm_min, norm_max)

    # WMAPE per feature on original scale data
    n_features = hist_data.shape[-1]
    wmape_per_feature = []

    for f in range(n_features):
        actual = hist_data[..., f].flatten()
        pred = reconstructed[..., f].flatten()
        abs_error = np.sum(np.abs(actual - pred))
        actual_sum = np.sum(np.abs(actual))
        if actual_sum > 0:
            wmape_f = abs_error / actual_sum
        else:
            wmape_f = 0.0
        wmape_per_feature.append(wmape_f)

    wmape = np.mean(wmape_per_feature)

    return wmape, wmape_per_feature
