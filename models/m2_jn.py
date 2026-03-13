"""M2 models for Self-Join and Binary-Join selectivity estimation.

Two model architectures extracted from the authors' code:
  spatial-embedding/modelsSJ/code_py/myModel_JN.py
- M2_DNN_JN: Dense-only model (from JN_2Input_DENSE2L_DENSE2L_DENSE2L)
- M2_CNN_JN: CNN model (from JN_2Input_CNN2L_Conc_noBN_DENSE2L, Dropout=0.3)
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers


# Extracted from the authors' code: myModel_JN.py - JN_2Input_DENSE2L_DENSE2L_DENSE2L
class M2_DNN_JN(Model):
    """Dense M2 model for join selectivity.

    Architecture: 2 Dense layers on each input -> concatenate -> 3 Dense layers -> output.
    Args:
        dimx, dimy: embedding spatial dimensions
        f1-f5: hidden layer sizes
    """

    def __init__(self, dimx, dimy, f1, f2, f3, f4, f5):
        super(M2_DNN_JN, self).__init__()
        self.hidden1 = keras.layers.Dense(f1, activation="relu")
        self.hidden2 = keras.layers.Dense(f2, activation="relu")
        self.hidden3 = keras.layers.Dense(f1, activation="relu")
        self.hidden4 = keras.layers.Dense(f2, activation="relu")
        self.hidden5 = keras.layers.Dense(f3, activation="relu")
        self.hidden6 = keras.layers.Dense(f4, activation="relu")
        self.hidden7 = keras.layers.Dense(f5, activation="relu")
        self.output_model = keras.layers.Dense(1, activation="linear")

    def call(self, inputs):
        dataA, dataB = inputs
        flatA = keras.layers.Flatten()(dataA)
        flatB = keras.layers.Flatten()(dataB)
        h1 = self.hidden1(flatA)
        h2 = self.hidden2(h1)
        h3 = self.hidden3(flatB)
        h4 = self.hidden4(h3)
        concat = keras.layers.concatenate([h2, h4])
        h5 = self.hidden5(concat)
        h6 = self.hidden6(h5)
        h7 = self.hidden7(h6)
        out = self.output_model(h7)
        return out


# Adapted from the authors' code: myModel_JN.py - JN_2Input_CNN2L_Conc_noBN_DENSE2L
# (modified: Dropout changed from 0.2 to 0.3; uses conditional MaxPool)
class M2_CNN_JN(Model):
    """CNN M2 model for join selectivity.

    Architecture: 2 CNN layers on input A -> concatenate with flattened B -> 2 Dense layers -> output.
    Uses Dropout=0.3 (different from RQ's 0.2).
    Args:
        dimx, dimy: embedding spatial dimensions
        f1-f4: filter sizes
    """

    def __init__(self, dimx, dimy, f1, f2, f3, f4):
        super(M2_CNN_JN, self).__init__()
        self.hidden1 = keras.layers.Conv2D(f1, (3, 3), activation='relu', padding='same', strides=2)
        self.hidden2 = keras.layers.Conv2D(f2, (3, 3), activation='relu', padding='same', strides=2)
        self.hidden2_mp = keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.hidden4 = keras.layers.Dense(f3, activation="relu")
        self.hidden5 = keras.layers.Dense(f4, activation="relu")
        self.output_model = keras.layers.Dense(1, activation="linear")

    def call(self, inputs):
        dataA, dataB = inputs
        x1 = self.hidden1(dataA)
        x1 = self.hidden2(x1)
        x1 = self.hidden2_mp(x1)
        flatB = keras.layers.Flatten()(x1)
        flatA = keras.layers.Flatten()(dataB)
        concat = keras.layers.concatenate([flatA, flatB])
        x = self.hidden4(concat)
        x = self.hidden5(x)
        x = keras.layers.Dropout(0.3)(x)
        out = self.output_model(x)
        return out
