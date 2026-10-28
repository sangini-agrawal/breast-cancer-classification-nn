"""Neural network architecture (Phase 4)."""
from tensorflow import keras

from . import config


def build_model(n_features: int) -> keras.Sequential:
    """Feed-forward neural network: 30 inputs -> 32 ReLU -> Dropout -> 16 ReLU -> 2 Softmax.

    * ReLU in the hidden layers lets the network learn non-linear patterns.
    * Dropout randomly switches off 20 % of neurons during training to reduce over-fitting.
    * Softmax in the output layer turns the two scores into probabilities that sum to 1
      (P(malignant) + P(benign) = 1), which is the correct pairing for
      sparse categorical cross-entropy loss.
    """
    h1, h2 = config.HIDDEN_UNITS
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(n_features,)),
            keras.layers.Dense(h1, activation="relu", name="hidden_1"),
            keras.layers.Dropout(config.DROPOUT_RATE, name="dropout"),
            keras.layers.Dense(h2, activation="relu", name="hidden_2"),
            keras.layers.Dense(2, activation="softmax", name="output"),
        ],
        name="breast_cancer_nn",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
