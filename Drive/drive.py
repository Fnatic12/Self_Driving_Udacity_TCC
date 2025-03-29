import argparse
import base64
import os
import shutil
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
import cv2
import csv
from datetime import datetime
from flask import Flask
from io import BytesIO
from PIL import Image
from keras.models import load_model
from keras.losses import MeanSquaredError

# Inicializando SocketIO e Flask
sio = socketio.Server()
app = Flask(__name__)
model = None
input_shape = (224, 224)  # default, será sobrescrito

# Limites de velocidade
MAX_SPEED = 20
MIN_SPEED = 10
speed_limit = MAX_SPEED

# Métricas
metrics_file = "drive_metrics.csv"
with open(metrics_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "steering_angle_pred", "steering_angle_real",
                     "throttle", "speed", "error", "response_time", "collision_flag", "distance", "average_speed"])

# Variáveis globais
prev_steering_angle = None
prev_speed = None
collision_count = 0
total_distance = 0
speed_sum = 0
speed_count = 0
start_time = None
prev_time = None

def preprocess_image(image_pil):
    global input_shape
    image = np.asarray(image_pil)
    image = cv2.resize(image, (input_shape[1], input_shape[0]))  # Resize para altura x largura
    image = image / 255.0
    return image

@sio.on('telemetry')
def telemetry(sid, data):
    global prev_steering_angle, prev_speed, collision_count, total_distance
    global speed_sum, speed_count, prev_time

    if data:
        current_time = time.time()
        if prev_time is None:
            prev_time = current_time

        # Dados do simulador
        steering_angle_real = float(data["steering_angle"])
        throttle = float(data["throttle"])
        speed = float(data["speed"])

        # Imagem base64 -> PIL
        img_string = data["image"]
        img_bytes = BytesIO(base64.b64decode(img_string))
        image = Image.open(img_bytes)

        try:
            image = preprocess_image(image)
            image = np.array([image])  # (1, H, W, 3)

            steering_angle_pred = float(model.predict(image, batch_size=1)[0][0])

            # Ajuste de velocidade
            global speed_limit
            speed_limit = MIN_SPEED if speed > speed_limit else MAX_SPEED
            throttle = 1.0 - (steering_angle_pred ** 2) - ((speed / speed_limit) ** 2)

            # Métricas
            response_time = time.time() - current_time
            error = abs(steering_angle_real - steering_angle_pred)
            time_diff = current_time - prev_time
            distance = (speed * 0.44704) * time_diff  # mph → m/s
            total_distance += distance
            prev_time = current_time

            speed_sum += speed
            speed_count += 1
            average_speed = speed_sum / speed_count if speed_count > 0 else 0

            collision_flag = 0
            if prev_speed is not None and speed < prev_speed * 0.5:
                collision_flag = 1
                collision_count += 1

            prev_speed = speed
            prev_steering_angle = steering_angle_pred

            print(f"🔁 Predição: Steering={steering_angle_pred:.4f} | Throttle={throttle:.4f} | Speed={speed:.2f} | Error={error:.4f} | Dist={total_distance:.2f}m | AvgSpeed={average_speed:.2f}mph | Collisions={collision_count}")

            with open(metrics_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.utcnow(), steering_angle_pred, steering_angle_real, throttle, speed, error, response_time, collision_flag, total_distance, average_speed])

            send_control(steering_angle_pred, throttle)

        except Exception as e:
            print(f"❌ Erro na predição: {e}")

@sio.on('connect')
def connect(sid, environ):
    print(f"✅ Cliente conectado: {sid}")
    for _ in range(5):
        send_control(0.0, 0.5)
        time.sleep(1)
    print("🚀 Teste de aceleração enviado.")

def send_control(steering_angle, throttle):
    sio.emit("steer", data={'steering_angle': str(steering_angle), 'throttle': str(throttle)}, skip_sid=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Caminho para o modelo treinado (ex: model.h5)")
    parser.add_argument("image_folder", type=str, nargs="?", default="", help="Pasta opcional para salvar imagens")
    args = parser.parse_args()

    print(f"📦 Carregando modelo: {args.model}")
    model = load_model(args.model, custom_objects={'mse': MeanSquaredError()})
    
    # Detectar shape de entrada automaticamente
    input_shape = model.input_shape[1:3]
    print(f"📐 Input shape do modelo detectado: {input_shape}")

    if args.image_folder:
        if os.path.exists(args.image_folder):
            shutil.rmtree(args.image_folder)
        os.makedirs(args.image_folder)

    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(("0.0.0.0", 4567)), app)