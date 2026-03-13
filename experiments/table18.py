"""Table 18: BJ selectivity - Full detailed results (all configs).

Runs all combinations: 2 AE x 5 DNN + 2 AE x 5 CNN = 20 combos per training type.

Output columns: Net_arch, Hyperpar, AE_S1_WMAPE, AE_S1_Time, AE_C2_WMAPE, ..., BL
"""
import os
import numpy as np
import pandas as pd

from training.train_m2 import create_m2_model, train_m2
from evaluation.metrics import compute_baseline_jn
import configs as cfg


def run(data_dir, output_dir, **kwargs):
    """Run Table 18 experiment - Full BJ selectivity scan."""
    print("\n" + "=" * 60)
    print("TABLE 18: BJ Selectivity - Full Configs")
    print("=" * 60)

    # Synthetic AEs
    ae_synth = ["AE_S1", "AE_C2"]
    # Real AEs
    ae_real = ["AE_S4", "AE_C3"]

    all_m2_configs = list(cfg.M2_DNN_CONFIGS.values()) + list(cfg.M2_CNN_CONFIGS.values())
    all_results = []

    for training_type, ae_names in [("synthetic", ae_synth), ("synthetic+real", ae_real)]:
        print(f"\n--- Training type: {training_type} ---")
        results_rows = []

        for m2_cfg in all_m2_configs:
            row = {
                'Net_arch': f"M2_{m2_cfg.m2_type.upper()}",
                'Hyperpar': m2_cfg.name,
            }

            for ae_name in ae_names:
                ae_cfg = cfg.AE_CONFIGS[ae_name]
                emb_shape = ae_cfg.emb_shape

                x_file = os.path.join(data_dir, f"x_bj_sel_{ae_name}.npy")
                x1_file = os.path.join(data_dir, f"x1_bj_sel_{ae_name}.npy")
                y_file = os.path.join(data_dir, f"y_bj_sel_{ae_name}.npy")

                if not os.path.exists(x_file):
                    print(f"  Warning: {x_file} not found, skipping {ae_name}")
                    row[f'{ae_name}_WMAPE'] = 'N/A'
                    row[f'{ae_name}_Time'] = 'N/A'
                    continue

                x = np.load(x_file)
                x1 = np.load(x1_file) if os.path.exists(x1_file) else np.zeros((x.shape[0], 1))
                y = np.load(y_file)

                print(f"  {ae_name} + {m2_cfg.name} ({m2_cfg.m2_type}): ", end="", flush=True)

                model = create_m2_model("bj", m2_cfg.m2_type, emb_shape, m2_cfg.filters)
                _, _, metrics, train_time = train_m2(model, x, x1, y)

                row[f'{ae_name}_WMAPE'] = f"{metrics['wmape_tot']:.4f}"
                row[f'{ae_name}_Time'] = f"{train_time:.1f}"
                print(f"WMAPE={metrics['wmape_tot']:.4f}, Time={train_time:.1f}s")

            # Baseline
            y_bl_file = os.path.join(data_dir, f"y_bj_sel_{ae_names[0]}.npy")
            if os.path.exists(y_bl_file):
                y_bl = np.load(y_bl_file)
                row['BL'] = f"{compute_baseline_jn(y_bl):.4f}"
            else:
                row['BL'] = 'N/A'

            results_rows.append(row)

        df = pd.DataFrame(results_rows)
        suffix = "synth" if training_type == "synthetic" else "real"
        output_file = os.path.join(output_dir, f"table18_{suffix}.csv")
        df.to_csv(output_file, index=False)
        all_results.append(df)
        print(f"\nResults saved to {output_file}")
        print(df.to_string(index=False))

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        output_file = os.path.join(output_dir, "table18.csv")
        combined.to_csv(output_file, index=False)
        return combined
    return pd.DataFrame()
