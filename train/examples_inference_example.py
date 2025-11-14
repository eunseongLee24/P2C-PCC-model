import numpy as np
from model.p2c_pcc import P2C_PCC
from train.train_p2c_pcc import build_p2c_pcc_model

# Example dummy input
x_dummy = np.random.rand(1, 10, 1).astype(np.float32)

model = build_p2c_pcc_model()
y_pred = model.predict(x_dummy)

print("Prediction:", y_pred)
