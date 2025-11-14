import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from model.p2c_pcc import P2C_PCC

def build_p2c_pcc_model(time_steps=10, feature_dim=1, filters=100):
    inputs = layers.Input(shape=(time_steps, feature_dim))
    x = P2C_PCC(filters=filters, kernel_size=3)(inputs)
    x = layers.Conv1D(filters=100, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.Conv1D(filters=100, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation='relu')(x)
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='mean_absolute_error',
        metrics=['mean_absolute_error']
    )
    return model

if __name__ == "__main__":
    model = build_p2c_pcc_model()
    model.summary()
