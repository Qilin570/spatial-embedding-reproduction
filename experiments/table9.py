"""Table 9: Binary Join MBR Tests - Best results (fixed hyperparameters from paper).

Reproduces the 8 rows of Table 9 (SJMR + DJ algorithms):
  SJMR: M2_dnn + AE_C2 + dH4, M2_cnn + AE_C2 + cH3
  DJ:   M2_dnn + AE_C2 + dH5, M2_cnn + AE_C2 + cH3
  SJMR: M2_dnn + AE_S4 + dH3, M2_cnn + AE_S4 + cH3
  DJ:   M2_dnn + AE_S4 + dH4, M2_cnn + AE_S4 + cH5

Output columns: Algorithm, M2_arch, Training, Autoencoder, Hyperpar, Time, WMAPE
"""
import os
import numpy as np
import pandas as pd

from training.train_m2 import create_m2_model, train_m2
import configs as cfg


def run(data_dir, output_dir, **kwargs):
    """Run Table 9 experiment with fixed hyperparameters from paper."""
    print("\n" + "=" * 60)
    print("TABLE 9: Binary Join MBR Tests - Best Results")
    print("=" * 60)

    # Fixed experiments from paper Table 9
    # (algorithm, m2_arch, m2_type, ae_name, m2_cfg_name)
    experiments = [
        ("SJMR", "M2_DNN", "dnn", "AE_C2", "dH4"),
        ("SJMR", "M2_CNN", "cnn", "AE_C2", "cH3"),
        ("DJ",   "M2_DNN", "dnn", "AE_C2", "dH5"),
        ("DJ",   "M2_CNN", "cnn", "AE_C2", "cH3"),
        ("SJMR", "M2_DNN", "dnn", "AE_S4", "dH3"),
        ("SJMR", "M2_CNN", "cnn", "AE_S4", "cH3"),
        ("DJ",   "M2_DNN", "dnn", "AE_S4", "dH4"),
        ("DJ",   "M2_CNN", "cnn", "AE_S4", "cH5"),
    ]

    results = []
    for algorithm, m2_arch, m2_type, ae_name, m2_cfg_name in experiments:
        ae_cfg = cfg.AE_CONFIGS[ae_name]
        emb_shape = ae_cfg.emb_shape

        if m2_type == "dnn":
            m2_cfg = cfg.M2_DNN_CONFIGS[m2_cfg_name]
        else:
            m2_cfg = cfg.M2_CNN_CONFIGS[m2_cfg_name]

        # BJ MBR data files
        x_file = os.path.join(data_dir, f"x_bj_mbr_{ae_name}.npy")
        x1_file = os.path.join(data_dir, f"x1_bj_mbr_{ae_name}.npy")
        y_file = os.path.join(data_dir, f"y_bj_mbr_{ae_name}.npy")

        if not os.path.exists(x_file) or not os.path.exists(y_file):
            print(f"  Skipping {algorithm} + {ae_name}: data not found")
            continue

        x = np.load(x_file)
        x1 = np.load(x1_file) if os.path.exists(x1_file) else np.zeros((x.shape[0], 1))
        y = np.load(y_file)

        print(f"\n  {algorithm} + {m2_arch} + {ae_name} + {m2_cfg_name} ({m2_cfg.filters})")
        model = create_m2_model("bj", m2_type, emb_shape, m2_cfg.filters)
        _, _, metrics, train_time = train_m2(model, x, x1, y)
        print(f"  WMAPE={metrics['wmape_tot']:.4f}")

        results.append({
            'Algorithm': algorithm,
            'M2_arch': m2_arch,
            'Training': ae_cfg.trained_on,
            'Autoencoder': ae_name,
            'Hyperpar': m2_cfg_name,
            'Time': f"{train_time:.1f}",
            'WMAPE': f"{metrics['wmape_tot']:.4f}",
        })

    df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, "table9.csv")
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    print(df.to_string(index=False))
    return df
