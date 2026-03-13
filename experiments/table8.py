"""Table 8: Binary Join selectivity - Best results.

Uses AE_C2 (synthetic) and AE_S4/C3 (real) with best DNN and CNN configs.

Output columns: M2_arch, Training, Autoencoder, Hyperpar, Time, WMAPE, Baseline
"""
import os
import numpy as np
import pandas as pd

from training.train_m2 import create_m2_model, train_m2
from evaluation.metrics import compute_baseline_jn
import configs as cfg


def run(data_dir, output_dir, **kwargs):
    """Run Table 8 experiment - BJ selectivity."""
    print("\n" + "=" * 60)
    print("TABLE 8: Binary Join Selectivity - Best Results")
    print("=" * 60)

    # Try to extract from Table 18 first
    table18_file = os.path.join(output_dir, "table18.csv")
    if os.path.exists(table18_file):
        print("Extracting best results from Table 18...")
        df18 = pd.read_csv(table18_file)
        results = _extract_best_from_table18(df18)
        if results:
            df = pd.DataFrame(results)
            output_file = os.path.join(output_dir, "table8.csv")
            df.to_csv(output_file, index=False)
            print(f"\nResults saved to {output_file}")
            print(df.to_string(index=False))
            return df

    experiments = [
        ("M2_CNN", "AE_C2", list(cfg.M2_CNN_CONFIGS.values())),
        ("M2_DNN", "AE_C2", list(cfg.M2_DNN_CONFIGS.values())),
        ("M2_CNN", "AE_S4", list(cfg.M2_CNN_CONFIGS.values())),
        ("M2_DNN", "AE_S4", list(cfg.M2_DNN_CONFIGS.values())),
        ("M2_CNN", "AE_C3", list(cfg.M2_CNN_CONFIGS.values())),
        ("M2_DNN", "AE_C3", list(cfg.M2_DNN_CONFIGS.values())),
    ]

    results = []
    for m2_arch, ae_name, m2_cfgs in experiments:
        ae_cfg = cfg.AE_CONFIGS[ae_name]
        emb_shape = ae_cfg.emb_shape
        m2_type = "cnn" if "CNN" in m2_arch else "dnn"

        x_file = os.path.join(data_dir, f"x_bj_sel_{ae_name}.npy")
        x1_file = os.path.join(data_dir, f"x1_bj_sel_{ae_name}.npy")
        y_file = os.path.join(data_dir, f"y_bj_sel_{ae_name}.npy")

        if not os.path.exists(x_file):
            print(f"  Skipping {ae_name}: data not found")
            continue

        x = np.load(x_file)
        x1 = np.load(x1_file) if os.path.exists(x1_file) else np.zeros((x.shape[0], 1))
        y = np.load(y_file)

        best_wmape = float('inf')
        best_result = None

        for m2_cfg in m2_cfgs:
            if m2_cfg.m2_type != m2_type:
                continue
            print(f"  {m2_arch} + {ae_name} + {m2_cfg.name}: ", end="", flush=True)
            model = create_m2_model("bj", m2_type, emb_shape, m2_cfg.filters)
            _, _, metrics, train_time = train_m2(model, x, x1, y)
            print(f"WMAPE={metrics['wmape_tot']:.4f}")

            if metrics['wmape_tot'] < best_wmape:
                best_wmape = metrics['wmape_tot']
                best_result = {
                    'M2_arch': m2_arch,
                    'Training': ae_cfg.trained_on,
                    'Autoencoder': ae_name,
                    'Hyperpar': m2_cfg.name,
                    'Time': f"{train_time:.1f}",
                    'WMAPE': f"{best_wmape:.4f}",
                    'Baseline': f"{compute_baseline_jn(y):.4f}",
                }

        if best_result:
            results.append(best_result)

    df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, "table8.csv")
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    print(df.to_string(index=False))
    return df


def _extract_best_from_table18(df18):
    """Extract best results per AE from Table 18."""
    results = []
    wmape_cols = [c for c in df18.columns if c.endswith('_WMAPE')]
    for col in wmape_cols:
        ae_name = col.replace('_WMAPE', '')
        try:
            values = pd.to_numeric(df18[col], errors='coerce')
            best_idx = values.idxmin()
            if pd.notna(best_idx):
                ae_cfg = cfg.AE_CONFIGS.get(ae_name)
                if ae_cfg:
                    time_col = f'{ae_name}_Time'
                    results.append({
                        'M2_arch': df18.loc[best_idx, 'Net_arch'] if 'Net_arch' in df18.columns else 'M2_CNN',
                        'Training': ae_cfg.trained_on,
                        'Autoencoder': ae_name,
                        'Hyperpar': df18.loc[best_idx, 'Hyperpar'],
                        'Time': df18.loc[best_idx, time_col] if time_col in df18.columns else 'N/A',
                        'WMAPE': f"{values[best_idx]:.4f}",
                        'Baseline': df18.loc[best_idx, 'BL'] if 'BL' in df18.columns else 'N/A',
                    })
        except (ValueError, KeyError):
            continue
    return results
