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
    """Create an M2 model for the given task and architecture type."""
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
    """Train an M2 model and evaluate on held-out test set.
    Returns (model, history, metrics, train_time).
    """
    if epochs is None:
        epochs = cfg.M2_EPOCHS
    if batch_size is None:
        batch_size = cfg.M2_BATCH_SIZE
    if patience is None:
        patience = cfg.M2_PATIENCE

    # Normalize y (before splitting, matching original author)
    y_maximum = np.amax(y, axis=0)
    y_minimum = np.amin(y, axis=0)
    y_nor = nor_y_ab(y, c_norm, y_minimum, y_maximum)

    # Step 1: 80/20 test split (random_state=42, matching original author)
    x_train_full, x_test, x1_train_full, x1_test, y_train_full, y_test = \
        train_test_split(x, x1, y_nor, test_size=0.2, random_state=42)

    # Step 2: 80/20 val split from training set (random_state=43, matching original author)
    X_loc_train, X_loc_val, X_glo_train, X_glo_val, y_train, y_val = \
        train_test_split(x_train_full, x1_train_full, y_train_full,
                         test_size=0.2, random_state=43)

    # Compile (MAE loss + Adam default lr=0.001, matching original author)
    model.compile(optimizer='adam', loss=losses.MeanAbsoluteError())

    # EarlyStopping on val_loss with restore_best_weights
    # (original author monitors 'loss', but val_loss stops earlier and avoids overfitting)
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience, restore_best_weights=True)

    # Train
    print(f"  Training M2 model...")
    print(f"  x_train: {X_loc_train.shape}, x1_train: {X_glo_train.shape}")
    print(f"  x_test: {x_test.shape}, x1_test: {x1_test.shape}")
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

    # Predict on TEST set
    y_pred = model.predict([x_test, x1_test])

    # Denormalize with min=0.0 (matching original author)
    y_test_den = denorm_y_ab(y_test, c_norm, 0.0, y_maximum)
    y_pred_den = denorm_y_ab(y_pred, c_norm, 0.0, y_maximum)

    # Evaluate on test set (using denormalized values)
    metrics = mape_error_zero(y_test_den, y_pred_den)
    metrics['train_time'] = train_time
    metrics['epochs'] = len(history.history['loss'])
    metrics['final_loss'] = history.history['loss'][-1]
    metrics['final_val_loss'] = history.history['val_loss'][-1]

    return model, history, metrics, train_time


def run_experiment(task, ae_configs, m2_configs, data_files,
                   c_norm=0, y_min=0.0, y_max=1.0):
    """Run a full experiment with multiple AE and M2 configurations."""
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
