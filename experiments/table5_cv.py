"""Table 5 (CV): RQ selectivity - 5-fold cross-validation.

This script mirrors `experiments/table5.py` (fixed hyperparameters from the
paper) but replaces the single train/eval split with 5-fold cross-validation.

Output columns:
  M2_arch, Training, Autoencoder, Hyperpar,
  Time_mean, Time_std, WMAPE_mean, WMAPE_std
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras import losses

import configs as cfg
from data.normalization import nor_y_ab, denorm_y_ab
from evaluation.metrics import mape_error_zero
from training.train_m2 import create_m2_model


@dataclass(frozen=True)
class _FoldResult:
    wmape_tot: float
    train_time: float
    epochs_ran: int


def _train_m2_fold(
    model: tf.keras.Model,
    *,
    x_train: np.ndarray,
    x1_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    x1_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    x1_test: np.ndarray,
    y_test: np.ndarray,
    c_norm: float,
    y_minimum: np.ndarray,
    y_maximum: np.ndarray,
    epochs: int,
    batch_size: int,
    patience: int,
) -> _FoldResult:
    """Train one fold and evaluate on the held-out fold."""

    # Match `training/train_m2.py` normalization behavior:
    # - Normalize with (c_norm, y_minimum, y_maximum) computed from the fold's
    #   training portion (outer CV train_idx).
    # - Denormalize predictions with min_val=0.0 and max_val=y_maximum.
    y_train_nor = nor_y_ab(y_train, c_norm, y_minimum, y_maximum)
    y_val_nor = nor_y_ab(y_val, c_norm, y_minimum, y_maximum)
    y_test_nor = nor_y_ab(y_test, c_norm, y_minimum, y_maximum)

    model.compile(optimizer="adam", loss=losses.MeanAbsoluteError())

    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )

    t0 = time.time()
    history = model.fit(
        [x_train, x1_train],
        y_train_nor,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[callback],
        validation_data=([x_val, x1_val], y_val_nor),
        verbose=0,
    )
    train_time = time.time() - t0

    y_pred_nor = model.predict([x_test, x1_test], verbose=0)

    y_test_den = denorm_y_ab(y_test_nor, c_norm, 0.0, y_maximum)
    y_pred_den = denorm_y_ab(y_pred_nor, c_norm, 0.0, y_maximum)

    metrics = mape_error_zero(y_test_den, y_pred_den)
    epochs_ran = len(history.history.get("loss", [])) or 0

    return _FoldResult(
        wmape_tot=float(metrics["wmape_tot"]),
        train_time=float(train_time),
        epochs_ran=int(epochs_ran),
    )


def run(
    data_dir: str,
    output_dir: str,
    *,
    n_splits: int = 5,
    shuffle: bool = True,
    seed: int = 42,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    patience: Optional[int] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Run 5-fold cross-validation version of Table 5."""

    print("\n" + "=" * 60)
    print("TABLE 5 (CV): RQ Selectivity - 5-fold Cross-Validation")
    print("=" * 60)
    print(f"n_splits={n_splits}, shuffle={shuffle}, seed={seed}")

    # Fixed experiments from paper Table 5
    experiments = [
        # (m2_arch, m2_type, ae_name, m2_cfg_name)
        ("M2_DNN", "dnn", "AE_S1", "dH3"),
        ("M2_CNN", "cnn", "AE_C2", "cH4"),
        ("M2_DNN", "dnn", "AE_S3", "dH2"),
        ("M2_CNN", "cnn", "AE_S4", "cH4"),
    ]

    epochs = int(epochs) if epochs is not None else int(cfg.M2_EPOCHS)
    batch_size = int(batch_size) if batch_size is not None else int(cfg.M2_BATCH_SIZE)
    patience = int(patience) if patience is not None else int(cfg.M2_PATIENCE)

    results: List[Dict[str, Any]] = []

    # Cache loaded arrays by AE name (x/x1/y are the same within a given AE).
    loaded: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    c_norm = 0.0

    for m2_arch, m2_type, ae_name, m2_cfg_name in experiments:
        ae_cfg = cfg.AE_CONFIGS[ae_name]
        emb_shape = ae_cfg.emb_shape

        if m2_type == "dnn":
            m2_cfg = cfg.M2_DNN_CONFIGS[m2_cfg_name]
        else:
            m2_cfg = cfg.M2_CNN_CONFIGS[m2_cfg_name]

        if ae_name not in loaded:
            x_file = os.path.join(data_dir, f"x_rq_{ae_name}.npy")
            x1_file = os.path.join(data_dir, f"x1_rq_{ae_name}.npy")
            y_file = os.path.join(data_dir, f"y_rq_{ae_name}.npy")

            if not os.path.exists(x_file) or not os.path.exists(y_file):
                print(f"  Skipping {ae_name}: missing input files.")
                continue

            x = np.load(x_file)
            x1 = np.load(x1_file) if os.path.exists(x1_file) else np.zeros((x.shape[0], 1))
            y = np.load(y_file)
            loaded[ae_name] = (x, x1, y)

        x, x1, y = loaded[ae_name]
        n_samples = int(x.shape[0])
        if n_samples < n_splits:
            print(
                f"  Skipping {ae_name}: not enough samples for {n_splits}-fold "
                f"(n_samples={n_samples})."
            )
            continue

        # For CV we compute normalization bounds per-fold using only y_train
        # (outer CV train_idx), so the outer test fold does not leak into scaling.

        print("\n" + "-" * 60)
        print(f"{m2_arch} + {ae_name} + {m2_cfg_name} (filters={m2_cfg.filters})")
        print(f"  samples={n_samples}, emb_shape={emb_shape}")
        print(f"  epochs={epochs}, batch_size={batch_size}, patience={patience}")

        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

        fold_wmapes: List[float] = []
        fold_times: List[float] = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(x), start=1):
            x_train = x[train_idx]
            x1_train = x1[train_idx]
            y_train = y[train_idx]

            x_test = x[test_idx]
            x1_test = x1[test_idx]
            y_test = y[test_idx]

            # Fold-specific normalization bounds (use only outer-CV training data).
            y_maximum = np.amax(y_train, axis=0)
            y_minimum = np.amin(y_train, axis=0)

            # Split the fold training portion into (train, val) for early stopping.
            # Mirror `training/train_m2.py` internal val split ratio (20% val).
            x_loc_tr, x_loc_val, x_glo_tr, x_glo_val, y_tr, y_val = train_test_split(
                x_train,
                x1_train,
                y_train,
                test_size=0.2,
                random_state=43,
            )

            model = create_m2_model("rq", m2_type, emb_shape, m2_cfg.filters)

            fold_result = _train_m2_fold(
                model,
                x_train=x_loc_tr,
                x1_train=x_glo_tr,
                y_train=y_tr,
                x_val=x_loc_val,
                x1_val=x_glo_val,
                y_val=y_val,
                x_test=x_test,
                x1_test=x1_test,
                y_test=y_test,
                c_norm=c_norm,
                y_minimum=y_minimum,
                y_maximum=y_maximum,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
            )

            fold_wmapes.append(fold_result.wmape_tot)
            fold_times.append(fold_result.train_time)

            print(f"  Fold {fold_idx}/{n_splits}: WMAPE={fold_result.wmape_tot:.4f}, time={fold_result.train_time:.1f}s")

        wmape_mean = float(np.mean(fold_wmapes))
        wmape_std = float(np.std(fold_wmapes, ddof=0))
        time_mean = float(np.mean(fold_times))
        time_std = float(np.std(fold_times, ddof=0))

        results.append(
            {
                "M2_arch": m2_arch,
                "Training": ae_cfg.trained_on,
                "Autoencoder": ae_name,
                "Hyperpar": m2_cfg_name,
                "Time_mean": f"{time_mean:.1f}",
                "Time_std": f"{time_std:.1f}",
                "WMAPE_mean": f"{wmape_mean:.4f}",
                "WMAPE_std": f"{wmape_std:.4f}",
            }
        )

        print(f"  CV result: WMAPE_mean={wmape_mean:.4f} (std={wmape_std:.4f})")

    df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, "table5_cv.csv")
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    print(df.to_string(index=False))
    return df

