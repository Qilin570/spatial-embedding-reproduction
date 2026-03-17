"""Table 6: Self-Join Selectivity - Best results (fixed hyperparameters from paper).

Reproduces the 4 rows of Table 6:
  M2_dnn + AE_C2 + dH3 (256,128,128,64,64)
  M2_cnn + AE_C2 + cH4 (512,256,256,128)
  M2_dnn + AE_S4 + dH1 (64,32,32,16,16)
  M2_cnn + AE_C3 + cH5 (1024,512,512,256)

Output columns: M2_arch, Training, Autoencoder, Hyperpar, Time, WMAPE
"""
import os
import numpy as np
import pandas as pd

from training.train_m2 import create_m2_model, train_m2
import configs as cfg


def run(data_dir, output_dir, **kwargs):
    """Run Table 6 experiment with fixed hyperparameters from paper."""
    print("\n" + "=" * 60)
    print("TABLE 6: Self-Join Selectivity - Best Results")
    print("=" * 60)

    # Fixed experiments from paper Table 6
    experiments = [
        # (m2_arch, m2_type, ae_name, m2_cfg_name)
        ("M2_DNN", "dnn", "AE_C2", "dH3"),
        ("M2_CNN", "cnn", "AE_C2", "cH4"),
        ("M2_DNN", "dnn", "AE_S4", "dH1"),
        ("M2_CNN", "cnn", "AE_C3", "cH5"),
    ]

    results = []
    for m2_arch, m2_type, ae_name, m2_cfg_name in experiments:
        ae_cfg = cfg.AE_CONFIGS[ae_name]
        emb_shape = ae_cfg.emb_shape

        if m2_type == "dnn":
            m2_cfg = cfg.M2_DNN_CONFIGS[m2_cfg_name]
        else:
            m2_cfg = cfg.M2_CNN_CONFIGS[m2_cfg_name]

        x_file = os.path.join(data_dir, f"x_sj_sel_{ae_name}.npy")
        x1_file = os.path.join(data_dir, f"x1_sj_sel_{ae_name}.npy")
        y_file = os.path.join(data_dir, f"y_sj_sel_{ae_name}.npy")

        if not os.path.exists(x_file):
            print(f"  Skipping {ae_name}: {x_file} not found")
            continue

        x = np.load(x_file)
        x1 = np.load(x1_file) if os.path.exists(x1_file) else np.zeros((x.shape[0], 1))
        y = np.load(y_file)

        print(f"\n  {m2_arch} + {ae_name} + {m2_cfg_name} ({m2_cfg.filters})")
        model = create_m2_model("sj", m2_type, emb_shape, m2_cfg.filters)
        _, _, metrics, train_time = train_m2(model, x, x1, y)
        print(f"  WMAPE={metrics['wmape_tot']:.4f}")

        results.append({
            'M2_arch': m2_arch,
            'Training': ae_cfg.trained_on,
            'Autoencoder': ae_name,
            'Hyperpar': m2_cfg_name,
            'Time': f"{train_time:.1f}",
            'WMAPE': f"{metrics['wmape_tot']:.4f}",
        })

    df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, "table6.csv")
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    print(df.to_string(index=False))
    return df
