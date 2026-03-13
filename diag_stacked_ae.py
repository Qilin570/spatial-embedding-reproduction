"""Quick diagnostic: ReLU vs LeakyReLU for AE_S4 only."""
import os, sys, numpy as np, tensorflow as tf
from tensorflow.keras import layers, losses, Model
from sklearn.model_selection import train_test_split
sys.path.insert(0, '/Users/yaming/Documents/python/spacial-embeddings/my-spatial-embedding')
from data.normalization import nor_g_ab, denorm_g_ab
import configs as cfg

DATA_DIR = '/Users/yaming/Documents/python/spacial-embeddings/my-spatial-embedding/downloaded_data'

class AE_diag(Model):
    def __init__(self, f1, f2, ld, activation='relu'):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(f1, activation=activation),
            layers.Dense(f2, activation=activation),
            layers.Dense(ld, activation=activation),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(f2, activation=activation),
            layers.Dense(f1, activation=activation),
            layers.Dense(128*128*6, activation='sigmoid'),
            layers.Reshape((128, 128, 6)),
        ])
    def call(self, x):
        return self.decoder(self.encoder(x))

def compute_wmape(model, hist_test, norm_min, norm_max):
    hist_norm, _, _ = nor_g_ab(hist_test.copy(), 1, norm_min, norm_max)
    rec_norm = model.predict(hist_norm, verbose=0)
    rec = denorm_g_ab(rec_norm, 1, norm_min, norm_max)
    wmapes = []
    for f in range(6):
        a = hist_test[..., f].flatten()
        p = rec[..., f].flatten()
        wmapes.append(np.sum(np.abs(a - p)) / np.sum(np.abs(a)) if np.sum(np.abs(a)) > 0 else 0)
    return np.mean(wmapes), wmapes

# Load data
hist_data = np.load(os.path.join(DATA_DIR, 'histograms_real.npy'))
_, norm_min, norm_max = nor_g_ab(hist_data.copy(), 1, -1, -1)
hist_train, hist_test = train_test_split(hist_data, test_size=0.2, random_state=42)
hist_norm, _, _ = nor_g_ab(hist_train.copy(), 1, norm_min, norm_max)
X_train, X_val = train_test_split(hist_norm, test_size=0.2, random_state=42)

print("=" * 70)
print("DIAGNOSTIC: AE_S4 (f1=16, f2=32, LD=384) - ReLU vs LeakyReLU")
print("Architecture: 98304 -> 16 -> 32 -> 384")
print("=" * 70)

for activation in ['relu', 'leaky_relu']:
    print(f"\n--- {activation.upper()} ---")
    for seed in [42, 123]:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        model = AE_diag(16, 32, 384, activation=activation)
        model.compile(optimizer='adam', loss=losses.MeanSquaredError())
        h = model.fit(X_train, X_train, batch_size=16, epochs=50,
                      validation_data=(X_val, X_val), verbose=0, shuffle=True)
        loss = h.history['loss'][-1]
        vloss = h.history['val_loss'][-1]
        wmape, pf = compute_wmape(model, hist_test, norm_min, norm_max)
        first_out = model.encoder.layers[1](model.encoder.layers[0](hist_norm[:50]))
        dead = int(np.sum(np.all(first_out.numpy() == 0, axis=0)))
        print(f"  seed={seed}: loss={loss:.6f}, val_loss={vloss:.6f}, "
              f"WMAPE={wmape:.4f}, dead_neurons={dead}/16")
        print(f"    per-feat: [{', '.join(f'{w:.2f}' for w in pf)}]")

print("\n--- AE_S3 (f1=16, f2=32, LD=48) ---")
for activation in ['relu', 'leaky_relu']:
    tf.random.set_seed(42)
    np.random.seed(42)
    model = AE_diag(16, 32, 48, activation=activation)
    model.compile(optimizer='adam', loss=losses.MeanSquaredError())
    h = model.fit(X_train, X_train, batch_size=16, epochs=50,
                  validation_data=(X_val, X_val), verbose=0, shuffle=True)
    loss = h.history['loss'][-1]
    vloss = h.history['val_loss'][-1]
    wmape, pf = compute_wmape(model, hist_test, norm_min, norm_max)
    first_out = model.encoder.layers[1](model.encoder.layers[0](hist_norm[:50]))
    dead = int(np.sum(np.all(first_out.numpy() == 0, axis=0)))
    print(f"  {activation:12s}: loss={loss:.6f}, val_loss={vloss:.6f}, "
          f"WMAPE={wmape:.4f}, dead={dead}/16")
