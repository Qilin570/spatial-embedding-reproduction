"""M2 models for Range Query selectivity estimation.

Two model architectures extracted from the authors' code:
  spatial-embedding/modelsRQ/code_py/myModel_RQ.py
- M2_DNN_RQ: Dense-only model (from RQ_sel_2Input_DENSE2L_DENSE3L)
- M2_CNN_RQ: CNN model (from RQ_sel_2Input_CNN2L_Conc_noBN_DENSE2L)
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers


# Extracted from the authors' code: myModel_RQ.py - RQ_sel_2Input_DENSE2L_DENSE3L
class M2_DNN_RQ(Model):
    """Dense M2 model for range query selectivity.

    Architecture: 2 Dense layers on input B -> concatenate with flattened A -> 3 Dense layers -> output.
    Args:
        dimx, dimy: embedding spatial dimensions
        f1-f5: hidden layer sizes (e.g., [64, 32, 32, 16, 16])
    """

    def __init__(self, dimx, dimy, f1, f2, f3, f4, f5):
        super(M2_DNN_RQ, self).__init__()
        self.hidden1 = keras.layers.Dense(f1, activation="relu")
        self.hidden2 = keras.layers.Dense(f2, activation="relu")
        self.hidden3 = keras.layers.Dense(f3, activation="relu")
        self.hidden4 = keras.layers.Dense(f4, activation="relu")
        self.hidden5 = keras.layers.Dense(f5, activation="relu")
        self.output_model = keras.layers.Dense(1, activation="linear")

    def call(self, inputs):
        dataA, dataB = inputs
        flatA = keras.layers.Flatten()(dataA)
        flatB = keras.layers.Flatten()(dataB)
        h1 = self.hidden1(flatB)
        h2 = self.hidden2(h1)
        concat = keras.layers.concatenate([flatA, h2])
        h3 = self.hidden3(concat)
        h4 = self.hidden4(h3)
        h5 = self.hidden5(h4)
        h6 = keras.layers.Dropout(0.2)(h5)
        out = self.output_model(h6)
        return out


# Adapted from the authors' code: myModel_RQ.py - RQ_sel_2Input_CNN2L_Conc_noBN_DENSE2L
# (modified: uses Conv2D+MaxPool instead of two separate Conv2D with conditional pooling)
class M2_CNN_RQ(Model):
    """CNN M2 model for range query selectivity.

    Architecture: 2 CNN layers on input A -> concatenate with flattened B -> 2 Dense layers -> output.
    Args:
        dimx, dimy: embedding spatial dimensions
        f1-f4: filter sizes (e.g., [64, 32, 32, 16])
    """

    def __init__(self, dimx, dimy, f1, f2, f3, f4):
        super(M2_CNN_RQ, self).__init__()
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
        if x1.shape[1] > 1:
            x1 = self.hidden2_mp(x1)
        flatB = keras.layers.Flatten()(x1)
        flatA = keras.layers.Flatten()(dataB)
        concat = keras.layers.concatenate([flatA, flatB])
        x = self.hidden4(concat)
        x = self.hidden5(x)
        x = keras.layers.Dropout(0.2)(x)
        out = self.output_model(x)
        return out
