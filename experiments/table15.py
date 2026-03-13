"""Table 15: RQ selectivity - Full detailed results (M2_CNN, all configs).

Runs all combinations of 4 AE configs x 5 CNN hyperparams for both
synthetic and real-trained AEs.

Output columns: Hyperpar, AE_S1_WMAPE, AE_S1_Time, AE_S2_WMAPE, ..., Baseline
"""
import os
import numpy as np
import pandas as pd
from tensorflow import keras

from training.train_m2 import create_m2_model, train_m2
from evaluation.metrics import compute_baseline_rq
import configs as cfg


def run(data_dir, output_dir, **kwargs):
    """Run Table 15 experiment - Full RQ selectivity scan."""
    print("\n" + "=" * 60)
    print("TABLE 15: RQ Selectivity - Full CNN Configs")
    print("=" * 60)

    # AE configs for synthetic training
    ae_synth = ["AE_S1", "AE_S2", "AE_C1", "AE_C2"]
    # AE configs for real training
    ae_real = ["AE_S3", "AE_S4", "AE_C3", "AE_C4"]

    cnn_configs = [cfg.M2_CNN_CONFIGS[f"cH{i}"] for i in range(1, 6)]

    all_results = []

    for training_type, ae_names in [("synthetic", ae_synth), ("synthetic+real", ae_real)]:
        print(f"\n--- Training type: {training_type} ---")
        results_rows = []

        for m2_cfg in cnn_configs:
            row = {'Hyperpar': m2_cfg.name}

            for ae_name in ae_names:
                ae_cfg = cfg.AE_CONFIGS[ae_name]
                emb_shape = ae_cfg.emb_shape

                # Load pre-generated input data
                x_file = os.path.join(data_dir, f"x_rq_{ae_name}.npy")
                x1_file = os.path.join(data_dir, f"x1_rq_{ae_name}.npy")
                y_file = os.path.join(data_dir, f"y_rq_{ae_name}.npy")

                if not os.path.exists(x_file):
                    print(f"  Warning: {x_file} not found, skipping {ae_name}")
                    row[f'{ae_name}_WMAPE'] = 'N/A'
                    row[f'{ae_name}_Time'] = 'N/A'
                    continue

                x = np.load(x_file)
                x1 = np.load(x1_file) if os.path.exists(x1_file) else np.zeros((x.shape[0], 1))
                y = np.load(y_file)

                print(f"  {ae_name} + {m2_cfg.name}: ", end="", flush=True)

                model = create_m2_model("rq", "cnn", emb_shape, m2_cfg.filters)
                _, _, metrics, train_time = train_m2(model, x, x1, y)

                row[f'{ae_name}_WMAPE'] = f"{metrics['wmape_tot']:.4f}"
                row[f'{ae_name}_Time'] = f"{train_time:.1f}"
                print(f"WMAPE={metrics['wmape_tot']:.4f}, Time={train_time:.1f}s")

            # Baseline
            if os.path.exists(os.path.join(data_dir, f"y_rq_{ae_names[0]}.npy")):
                y_bl = np.load(os.path.join(data_dir, f"y_rq_{ae_names[0]}.npy"))
                row['Baseline'] = f"{compute_baseline_rq(y_bl):.4f}"
            else:
                row['Baseline'] = 'N/A'

            results_rows.append(row)

        df = pd.DataFrame(results_rows)
        suffix = "synth" if training_type == "synthetic" else "real"
        output_file = os.path.join(output_dir, f"table15_{suffix}.csv")
        df.to_csv(output_file, index=False)
        all_results.append(df)
        print(f"\nResults saved to {output_file}")
        print(df.to_string(index=False))

    # Combined output
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        output_file = os.path.join(output_dir, "table15.csv")
        combined.to_csv(output_file, index=False)
        return combined
    return pd.DataFrame()
