"""Table 3: Exp2_M1 - CNN Autoencoder evaluation on synthetic data only.

Tests CNN autoencoders (AE_C1, AE_C2) trained on synthetic histograms.
Evaluates reconstruction quality with WMAPE on synthetic and real data.

Output columns: Autoencoder, LD, Hyperpar, Time, LOSS, VAL_LOSS, WMAPE, WMAPE_REAL
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from models.autoencoders import create_autoencoder
from training.train_ae import train_autoencoder, evaluate_autoencoder
from data.normalization import nor_g_ab
from data.histograms import gen_input_from_file
import configs as cfg


def _load_real_only(data_dir):
    """Extract pure real data from the combined dataset.

    The original authors' file histograms_loc_tot_real.npy contains
    real_data (first N rows) + synthetic_data (remaining rows).
    We extract the real-only portion by subtracting synthetic count.
    """
    combined_file = os.path.join(data_dir, "histograms_real.npy")
    synth_file = os.path.join(data_dir, "histograms_synthetic.npy")
    if not os.path.exists(combined_file) or not os.path.exists(synth_file):
        return None
    combined = np.load(combined_file)
    synth = np.load(synth_file)
    n_real = combined.shape[0] - synth.shape[0]
    if n_real <= 0:
        return None
    real_only = combined[:n_real]
    print(f"  Extracted {n_real} pure real histograms from combined dataset ({combined.shape[0]})")
    return real_only


def run(data_dir, output_dir, **kwargs):
    """Run Table 3 experiment.

    Args:
        data_dir: directory containing data files
        output_dir: directory to save results
    """
    print("\n" + "=" * 60)
    print("TABLE 3: CNN Autoencoder (Synthetic Data Only)")
    print("=" * 60)

    ae_names = ["AE_C1", "AE_C2"]
    results = []

    # Load synthetic histograms
    hist_file = os.path.join(data_dir, "histograms_synthetic.npy")
    if os.path.exists(hist_file):
        hist_synthetic = np.load(hist_file)
    else:
        # Try to generate from CSV files
        hist_dir = os.path.join(data_dir, "histograms", "new_datasets")
        summary_file = os.path.join(data_dir, "dataset-summaries.csv")
        if os.path.exists(hist_dir) and os.path.exists(summary_file):
            hist_synthetic, hist_global = gen_input_from_file(
                cfg.DIM_H_X, cfg.DIM_H_Y, cfg.DIM_H_Z,
                hist_dir, summary_file, 1, ""
            )
        else:
            print("ERROR: Cannot find synthetic histogram data.")
            print(f"  Tried: {hist_file}")
            print(f"  Tried: {hist_dir}")
            return pd.DataFrame()

    # Load pure real histograms for WMAPE_REAL evaluation
    # Note: histograms_real.npy is actually the combined dataset (real+synthetic).
    # The pure real data is the first N rows where N = combined - synthetic count.
    hist_real = _load_real_only(data_dir)

    # Compute data-derived normalization min/max from FULL synthetic dataset
    # Adapted from the authors' code: nor_g_ab(a_tot, 1, -1, -1)
    _, norm_min, norm_max = nor_g_ab(hist_synthetic.copy(), 1, -1, -1)
    print(f"Data-derived norm_min: {norm_min}")
    print(f"Data-derived norm_max: {norm_max}")

    # 80/20 split on synthetic data: train on 80%, evaluate on 20%
    hist_train, hist_test = train_test_split(
        hist_synthetic, test_size=0.2, random_state=42
    )
    print(f"Synthetic data split: {hist_train.shape[0]} train, {hist_test.shape[0]} test")

    for ae_name in ae_names:
        ae_cfg = cfg.AE_CONFIGS[ae_name]
        print(f"\n--- {ae_name}: {ae_cfg.ae_type}, LD={ae_cfg.latent_dim}, "
              f"f1={ae_cfg.f1}, f2={ae_cfg.f2} ---")

        # Check for pre-trained model
        model_path = os.path.join(data_dir, "model", ae_cfg.model_filename)
        loaded_from_file = False
        if os.path.exists(model_path):
            print(f"  Loading pre-trained model from {model_path}")
            from tensorflow import keras
            model = keras.models.load_model(model_path)
            loaded_from_file = True
            train_time = 0
            final_loss = 0
            final_val_loss = 0
        else:
            # Train on 80% training set with data-derived norm values
            model, history, train_time, _, _ = train_autoencoder(
                ae_cfg, hist_train, norm_min=norm_min, norm_max=norm_max
            )
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]

            # Save model
            model_save_dir = os.path.join(output_dir, "models")
            os.makedirs(model_save_dir, exist_ok=True)
            model.save(os.path.join(model_save_dir, ae_cfg.model_filename))

        # Evaluate on 20% test set with same normalization.
        # For models loaded from SavedModel, use encoder+decoder separately
        # to avoid TF2 traced call() graph issues with Dense autoencoders.
        wmape_synth, wmape_per_feat = evaluate_autoencoder(
            model, hist_test, norm_min, norm_max,
            use_encoder_decoder=loaded_from_file
        )

        # Evaluate on pure real data (using synthetic norm values, same as original code)
        wmape_real = 0.0
        if hist_real is not None:
            wmape_real, _ = evaluate_autoencoder(
                model, hist_real, norm_min, norm_max,
                use_encoder_decoder=loaded_from_file
            )

        result = {
            'Autoencoder': ae_name,
            'LD': ae_cfg.latent_dim,
            'Hyperpar': f"f1={ae_cfg.f1},f2={ae_cfg.f2}",
            'Time': f"{train_time:.1f}",
            'LOSS': f"{final_loss:.6f}",
            'VAL_LOSS': f"{final_val_loss:.6f}",
            'WMAPE': f"{wmape_synth:.4f}",
            'WMAPE_REAL': f"{wmape_real:.4f}",
        }
        results.append(result)
        print(f"  WMAPE (synthetic): {wmape_synth:.4f}")
        print(f"  WMAPE (real): {wmape_real:.4f}")

    df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, "table3.csv")
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    print(df.to_string(index=False))
    return df
