## Table 5 Cross-Validation (Table 51) Run Log

```text
############################################################
# Running Table 51
############################################################

============================================================
TABLE 5 (CV): RQ Selectivity - 5-fold Cross-Validation
============================================================
n_splits=5, shuffle=True, seed=42

------------------------------------------------------------
M2_DNN + AE_S1 + dH3 (filters=[256, 128, 128, 64, 64])
  samples=63410, emb_shape=(16, 8, 3)
  epochs=80, batch_size=32, patience=6
  Fold 1/5: WMAPE=0.0815, time=58.0s
  Fold 2/5: WMAPE=0.0829, time=57.3s
  Fold 3/5: WMAPE=0.0930, time=31.9s
  Fold 4/5: WMAPE=0.0847, time=48.8s
  Fold 5/5: WMAPE=0.0938, time=31.6s
  CV result: WMAPE_mean=0.0872 (std=0.0052)

------------------------------------------------------------
M2_CNN + AE_C2 + cH4 (filters=[512, 256, 256, 128])
  samples=63732, emb_shape=(32, 32, 3)
  epochs=80, batch_size=32, patience=6
  Fold 1/5: WMAPE=0.0802, time=2947.9s
  Fold 2/5: WMAPE=0.0766, time=3121.4s
  Fold 3/5: WMAPE=0.0884, time=1776.3s
  Fold 4/5: WMAPE=0.0762, time=3591.4s
  Fold 5/5: WMAPE=0.0857, time=2288.1s
  CV result: WMAPE_mean=0.0814 (std=0.0049)

------------------------------------------------------------
M2_DNN + AE_S3 + dH2 (filters=[128, 64, 64, 32, 32])
  samples=33554, emb_shape=(4, 4, 3)
  epochs=80, batch_size=32, patience=6
  Fold 1/5: WMAPE=0.3403, time=12.8s
  Fold 2/5: WMAPE=0.3666, time=10.4s
  Fold 3/5: WMAPE=0.3661, time=9.3s
  Fold 4/5: WMAPE=0.3397, time=16.0s
  Fold 5/5: WMAPE=0.3498, time=14.8s
  CV result: WMAPE_mean=0.3525 (std=0.0118)

------------------------------------------------------------
M2_CNN + AE_S4 + cH4 (filters=[512, 256, 256, 128])
  samples=33554, emb_shape=(16, 8, 3)
  epochs=80, batch_size=32, patience=6
  Fold 1/5: WMAPE=0.3155, time=603.2s
  Fold 2/5: WMAPE=0.3413, time=456.9s
  Fold 3/5: WMAPE=0.3256, time=612.4s
  Fold 4/5: WMAPE=0.3497, time=265.4s
  Fold 5/5: WMAPE=0.3129, time=661.3s
  CV result: WMAPE_mean=0.3290 (std=0.0144)

Results saved
M2_arch       Training Autoencoder Hyperpar Time_mean Time_std WMAPE_mean WMAPE_std
 M2_DNN      synthetic       AE_S1      dH3      45.5     11.7     0.0872    0.0052
 M2_CNN      synthetic       AE_C2      cH4    2745.0    639.7     0.0814    0.0049
 M2_DNN synthetic+real       AE_S3      dH2      12.6      2.5     0.3525    0.0118
 M2_CNN synthetic+real       AE_S4      cH4     519.8    144.4     0.3290    0.0144

Table 51 completed in 16669.7s
```

