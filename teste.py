from keras.models import load_model
import keras.losses
import cv2
import numpy as np

keras.losses.mse = keras.losses.MeanSquaredError()

model_path = "/Users/victormilani/udacity_tcc/model.h5"
model = load_model(model_path, custom_objects={'mse': keras.losses.mse})

image_path = "/Users/victormilani/data_udacity/IMG/center_2025_03_16_23_18_28_794.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Erro: Não foi possível carregar a imagem em {image_path}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = cv2.resize(image, (200, 66))

image = image / 127.5 - 1.0

steering_angle = model.predict(np.expand_dims(image, axis=0))

print(f"🔮 Predição do modelo (Ângulo de direção): {steering_angle[0][0]:.4f}°")