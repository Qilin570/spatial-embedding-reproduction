"""Autoencoder architectures for spatial histogram embeddings.

6 model classes from the original code:
- AutoencoderCNN_local: CNN AE for local histograms
- AutoencoderCNNDense_local: CNN+Dense hybrid AE
- Autoencoder_local: Dense (stacked) AE for local histograms
- Autoencoder_global: Dense AE for global histograms
- AutoencoderCNN_global: CNN AE for global histograms
- create_autoencoder: factory function

All model classes extracted from the authors' code:
  spatial-embedding/autoEncoders/code_py/myAutoencoder.py
"""
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers


# Extracted from the authors' code: myAutoencoder.py - AutoencoderCNN_local
class AutoencoderCNN_local(Model):
    """CNN autoencoder for local histograms (128x128x6 -> latent_dim)."""

    def __init__(self, latent_dim, dimx, dimy, dimz, f1, f2):
        super(AutoencoderCNN_local, self).__init__()
        self.latent_dim = latent_dim
        self.dimx = dimx
        self.dimy = dimy
        self.dimz = dimz
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(dimx, dimy, dimz)),
            layers.Conv2D(f1, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(f2, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(int(latent_dim / (dimx / 8 * dimy / 8)), (3, 3),
                          activation='relu', padding='same', strides=2),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(int(latent_dim / (dimx / 8 * dimy / 8)),
                                   kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(f2, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(f1, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(dimz, kernel_size=3, activation='sigmoid', padding='same'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Extracted from the authors' code: myAutoencoder.py - AutoencoderCNNDense_local
class AutoencoderCNNDense_local(Model):
    """CNN + Dense hybrid autoencoder for local histograms."""

    def __init__(self, latent_dim, dimx, dimy, dimz, f1, f2):
        super(AutoencoderCNNDense_local, self).__init__()
        self.latent_dim = latent_dim
        self.dimx = dimx
        self.dimy = dimy
        self.dimz = dimz
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(dimx, dimy, dimz)),
            layers.Conv2D(f1, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(f2, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(int(latent_dim / (16 * 16)), (3, 3),
                          activation='relu', padding='same', strides=2),
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(int(dimx / 4) * int(dimy / 4) * int(latent_dim / (16 * 16)),
                         activation='relu'),
            layers.Reshape((int(dimx / 4), int(dimy / 4), int(latent_dim / (16 * 16)))),
            layers.Conv2DTranspose(f2, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(f1, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(dimz, kernel_size=3, activation='sigmoid', padding='same'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Extracted from the authors' code: myAutoencoder.py - Autoencoder_local
class Autoencoder_local(Model):
    """Dense (stacked) autoencoder for local histograms."""

    def __init__(self, f1, f2, latent_dim, dimx, dimy, dimz):
        super(Autoencoder_local, self).__init__()
        self.latent_dim = latent_dim
        self.dimx = dimx
        self.dimy = dimy
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(f1, activation='relu'),
            layers.Dense(f2, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
        ])
        if dimz == 1:
            self.decoder = tf.keras.Sequential([
                layers.Dense(f2, activation='relu'),
                layers.Dense(f1, activation='relu'),
                layers.Dense(dimx * dimy * dimz, activation='sigmoid'),
                layers.Reshape((dimx, dimy)),
            ])
        else:
            self.decoder = tf.keras.Sequential([
                layers.Dense(f2, activation='relu'),
                layers.Dense(f1, activation='relu'),
                layers.Dense(dimx * dimy * dimz, activation='sigmoid'),
                layers.Reshape((dimx, dimy, dimz)),
            ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Extracted from the authors' code: myAutoencoder.py - Autoencoder_global
class Autoencoder_global(Model):
    """Dense autoencoder for global histograms (2D input)."""

    def __init__(self, latent_dim, dimx, dimy):
        super(Autoencoder_global, self).__init__()
        self.latent_dim = latent_dim
        self.dimx = dimx
        self.dimy = dimy
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(dimx * dimy, activation='sigmoid'),
            layers.Reshape((dimx, dimy)),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Extracted from the authors' code: myAutoencoder.py - AutoencoderCNN_global
class AutoencoderCNN_global(Model):
    """CNN autoencoder for global histograms."""

    def __init__(self, latent_dim, dimx, dimy):
        super(AutoencoderCNN_global, self).__init__()
        self.latent_dim = latent_dim
        self.dimx = dimx
        self.dimy = dimy
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(dimx, dimy, 1)),
            layers.Conv2D(latent_dim * 4, (3, 3), activation='relu', padding='same', strides=2),
            layers.Conv2D(latent_dim * 2, (3, 3), activation='relu', padding='same', strides=2),
            layers.Flatten(),
            layers.Dense(int(dimx / 4 * dimy / 4 * latent_dim), activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(int(dimx / 4 * dimy / 4 * latent_dim), activation='relu'),
            layers.Reshape((int(dimx / 4), int(dimy / 4), latent_dim)),
            layers.Conv2DTranspose(latent_dim * 2, kernel_size=3, strides=2,
                                   activation='relu', padding='same'),
            layers.Conv2DTranspose(latent_dim * 4, kernel_size=3, strides=2,
                                   activation='relu', padding='same'),
            layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def create_autoencoder(config, dimx=128, dimy=128, dimz=6):
    """Factory function to create an autoencoder from config.

    Args:
        config: AutoencoderConfig instance
        dimx, dimy, dimz: histogram dimensions
    Returns:
        autoencoder model
    """
    if config.ae_type == "cnn":
        return AutoencoderCNN_local(config.latent_dim, dimx, dimy, dimz,
                                    config.f1, config.f2)
    elif config.ae_type == "stacked":
        return Autoencoder_local(config.f1, config.f2, config.latent_dim,
                                 dimx, dimy, dimz)
    else:
        raise ValueError(f"Unknown ae_type: {config.ae_type}")


def create_global_autoencoder(latent_dim=2048, dimx=128, dimy=128, use_cnn=True):
    """Create a global histogram autoencoder.

    Args:
        latent_dim: embedding dimension
        dimx, dimy: histogram dimensions
        use_cnn: if True use CNN, else Dense
    Returns:
        autoencoder model
    """
    if use_cnn:
        return AutoencoderCNN_global(latent_dim, dimx, dimy)
    else:
        return Autoencoder_global(latent_dim, dimx, dimy)
