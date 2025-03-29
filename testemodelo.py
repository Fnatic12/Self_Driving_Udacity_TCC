from keras.models import load_model
from keras.losses import MeanSquaredError  # Necessário para compatibilidade
import tensorflow as tf

model = load_model("model1.h5", compile=False)

model.summary()
print("Saídas do modelo:", [output.name for output in model.outputs])