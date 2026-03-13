"""M2 model training for selectivity estimation.

Training logic adapted from the authors' code:
  spatial-embedding/modelsRQ/code_py/run_model_all.py
  spatial-embedding/modelsSJ/code_py/run_model_all.py
"""
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import losses
from sklearn.model_selection import train_test_split

from models.m2_rq import M2_DNN_RQ, M2_CNN_RQ
from models.m2_jn import M2_DNN_JN, M2_CNN_JN
from data.normalization import nor_y_ab, denorm_y_ab
from evaluation.metrics import mape_error_zero
import configs as cfg


def create_m2_model(task, m2_type, emb_shape, filters):
    """Create an M2 model.

    Args:
        task: "rq", "sj", or "bj"
        m2_type: "dnn" or "cnn"
        emb_shape: tuple (dim_e_x, dim_e_y, dim_e_z)
        filters: list of filter sizes
    Returns:
        model instance
    """
    dim_e_x, dim_e_y, _ = emb_shape

    if task == "rq":
        if m2_type == "dnn":
            return M2_DNN_RQ(dim_e_x, dim_e_y, *filters)
        else:
            return M2_CNN_RQ(dim_e_x, dim_e_y, *filters)
    else:  # sj or bj
        if m2_type == "dnn":
            return M2_DNN_JN(dim_e_x, dim_e_y, *filters)
        else:
            return M2_CNN_JN(dim_e_x, dim_e_y, *filters)


# Adapted from the authors' code: run_model_all.py
# (modified: generalized for both RQ and JN tasks, config-driven)
def train_m2(model, x, x1, y,
             epochs=None, batch_size=None, patience=None,
             c_norm=0, y_min=0.0, y_max=1.0):
    """Train an M2 model.

    Args:
        model: M2 model instance
        x: input embeddings (local)
        x1: input embeddings (global/query)
        y: target values
        epochs, batch_size, patience: training params
        c_norm: normalization constant for y
        y_min, y_max: min/max for y normalization
    Returns:
        model: trained model
        history: training history
        metrics: dict with evaluation metrics
        train_time: time in seconds
    """
    if epochs is None:
        epochs = cfg.M2_EPOCHS
    if batch_size is None:
        batch_size = cfg.M2_BATCH_SIZE
    if patience is None:
        patience = cfg.M2_PATIENCE

    # Normalize y
    y_maximum = np.amax(y, axis=0)
    y_minimum = np.amin(y, axis=0)
    y_nor = nor_y_ab(y, c_norm, y_minimum, y_maximum)

    # Split
    if x1.ndim > 1 and x1.shape[0] == x.shape[0]:
        X_loc_train, X_loc_val, X_glo_train, X_glo_val, y_train, y_val = \
            train_test_split(x, x1, y_nor, test_size=cfg.M2_VALIDATION_SPLIT, random_state=43)
    else:
        X_train_full, X_val_full, y_train, y_val = \
            train_test_split(x, y_nor, test_size=cfg.M2_VALIDATION_SPLIT, random_state=43)
        X_loc_train = X_train_full[:, :, :, 0:3]
        X_loc_val = X_val_full[:, :, :, 0:3]
        X_glo_train = X_train_full[:, :, :, 3:]
        X_glo_val = X_val_full[:, :, :, 3:]

    # Compile
    model.compile(optimizer='adam', loss=losses.MeanAbsoluteError())

    # Callbacks
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)

    # Train
    print(f"  Training M2 model...")
    print(f"  x_train: {X_loc_train.shape}, x1_train: {X_glo_train.shape}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}")

    t0 = time.time()
    history = model.fit(
        [X_loc_train, X_glo_train], y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[callback],
        validation_data=([X_loc_val, X_glo_val], y_val),
    )
    train_time = time.time() - t0

    # Predict on validation set
    y_pred = model.predict([X_loc_val, X_glo_val])

    # Denormalize
    y_val_den = denorm_y_ab(y_val, c_norm, y_minimum, y_maximum)
    y_pred_den = denorm_y_ab(y_pred, c_norm, y_minimum, y_maximum)

    # Evaluate
    metrics = mape_error_zero(y_val_den, y_pred_den)
    metrics['train_time'] = train_time
    metrics['epochs'] = len(history.history['loss'])
    metrics['final_loss'] = history.history['loss'][-1]
    metrics['final_val_loss'] = history.history['val_loss'][-1]

    return model, history, metrics, train_time


def run_experiment(task, ae_configs, m2_configs, data_files,
                   c_norm=0, y_min=0.0, y_max=1.0):
    """Run a full experiment with multiple AE and M2 configurations.

    Args:
        task: "rq", "sj", or "bj"
        ae_configs: list of (ae_name, AutoencoderConfig) to use
        m2_configs: list of M2HyperparamConfig to try
        data_files: dict mapping ae_name -> (x_file, x1_file, y_file, ds_file)
        c_norm: normalization constant
    Returns:
        DataFrame with all results
    """
    results = []

    for ae_name, ae_cfg in ae_configs:
        x_file, x1_file, y_file, ds_file = data_files[ae_name]

        print(f"\n{'='*60}")
        print(f"AE: {ae_name} ({ae_cfg.ae_type}, LD={ae_cfg.latent_dim})")
        print(f"{'='*60}")

        x = np.load(x_file)
        x1 = np.load(x1_file) if x1_file else np.zeros((x.shape[0], 1))
        y = np.load(y_file)

        emb_shape = ae_cfg.emb_shape

        for m2_cfg in m2_configs:
            print(f"\n  M2: {m2_cfg.name} ({m2_cfg.m2_type}), filters={m2_cfg.filters}")

            model = create_m2_model(task, m2_cfg.m2_type, emb_shape, m2_cfg.filters)
            _, _, metrics, train_time = train_m2(
                model, x, x1, y,
                c_norm=c_norm, y_min=y_min, y_max=y_max
            )

            result = {
                'ae_name': ae_name,
                'ae_type': ae_cfg.ae_type,
                'latent_dim': ae_cfg.latent_dim,
                'trained_on': ae_cfg.trained_on,
                'm2_name': m2_cfg.name,
                'm2_type': m2_cfg.m2_type,
                'filters': str(m2_cfg.filters),
                'time': train_time,
                'wmape': metrics['wmape'],
                'wmape_tot': metrics['wmape_tot'],
                'mape': metrics['mape'],
                'rma': metrics['rma'],
                'mae_zero': metrics['mae_zero'],
                'epochs': metrics['epochs'],
                'final_loss': metrics['final_loss'],
            }
            results.append(result)
            print(f"  -> WMAPE={metrics['wmape']:.4f}, WMAPE_TOT={metrics['wmape_tot']:.4f}, "
                  f"Time={train_time:.1f}s")

    return pd.DataFrame(results)
