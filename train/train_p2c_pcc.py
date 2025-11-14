import tensorflow as tf
from tensorflow.keras import layers, models
from model.p2c_pcc import P2C_PCC


def build_p2c_pcc_model(time_steps=10, filters=100):
    """
    Multi-input P2C-PCC model (9 parallel inputs)
    Structure identical to paper:
    - 9 branches (each: time_steps × 1)
    - Each branch → P2C-PCC
    - Concatenate all P2C-PCC outputs
    - 3-layer Conv1D stack
    - Dense(100) → Dense(1)
    """

    inputs = []
    p2c_outputs = []

    # ---- 9 parallel inputs ----
    for i in range(9):
        inp = layers.Input(shape=(time_steps, 1), name=f"input_{i+1}")
        inputs.append(inp)

        # apply P2C-PCC to each input branch
        x = P2C_PCC(filters=filters, kernel_size=3)(inp)
        p2c_outputs.append(x)

    # ---- Concatenate all P2C-PCC outputs ----
    x = layers.Concatenate(name="concat_p2c")(p2c_outputs)

    # ---- CNN blocks (3 layers, same as your original code) ----
    x = layers.Conv1D(filters=100, kernel_size=3, activation='relu',
                      padding='same', name="conv1")(x)
    x = layers.Conv1D(filters=100, kernel_size=3, activation='relu',
                      padding='same', name="conv2")(x)
    x = layers.Conv1D(filters=100, kernel_size=3, activation='relu',
                      padding='same', name="conv3")(x)

    # ---- Dense layers ----
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(100, activation='relu', name="dense_hidden")(x)
    output = layers.Dense(1, name="output")(x)

    # ---- Build model ----
    model = models.Model(inputs=inputs, outputs=output, name="P2C_PCC_Model")

    # ---- Compile ----
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_absolute_error',
        metrics=['mean_absolute_error']
    )

    return model


if __name__ == "__main__":
    model = build_p2c_pcc_model()
    model.summary()
