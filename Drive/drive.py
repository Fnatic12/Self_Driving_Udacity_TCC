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

##python drive.py model.h5##

# Inicializando o servidor SocketIO e Flask
sio = socketio.Server()
app = Flask(__name__)

# Inicializando o modelo e variáveis globais
model = None
prev_image_array = None

# Definição de velocidade máxima e mínima
MAX_SPEED = 25
MAX_SPEED = 20
MIN_SPEED = 10
speed_limit = MAX_SPEED

# Criando arquivo CSV para armazenar métricas
metrics_file = "drive_metrics.csv"
with open(metrics_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "steering_angle_pred", "steering_angle_real", 
                    "throttle", "speed", "error", "response_time", "collision_flag", "distance", "average_speed"])

# Variáveis para análise de métricas
prev_steering_angle = None
prev_speed = None
collision_count = 0
total_distance = 0
speed_sum = 0
speed_count = 0
start_time = None
prev_time = None

def preprocess_image(image):
    """
    Pré-processa a imagem antes de ser enviada para o modelo.
    """
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2YUV)  # Converte RGB -> YUV
    image = cv2.resize(image, (200, 66))  # Redimensiona para o tamanho esperado pelo modelo
    image = image / 255.0  # Normaliza os valores da imagem
    return image

@sio.on('telemetry')
def telemetry(sid, data):
    """
    Função que recebe dados do simulador e envia comandos de direção.
    """
    global prev_steering_angle, prev_speed, collision_count, total_distance, speed_sum, speed_count, prev_time

    if data:
        current_time = time.time()
        if prev_time is None:
            prev_time = current_time  # Define o tempo inicial na primeira iteração

        # Captura os dados enviados pelo simulador
        steering_angle_real = float(data["steering_angle"])  # Direção atual
        throttle = float(data["throttle"])  # Aceleração real
        speed = float(data["speed"])  # Velocidade real

        # Decodifica a imagem da câmera
        img_string = data["image"]
        img_bytes = BytesIO(base64.b64decode(img_string))
        image = Image.open(img_bytes)

        try:
            # Processa a imagem antes de enviar para o modelo
            image = preprocess_image(image)
            image = np.array([image])  # Modelo espera um array 4D

            # Faz a predição do ângulo de direção
            steering_angle_pred = float(model.predict(image, batch_size=1)[0][0])

            # Ajusta a aceleração conforme a velocidade do carro
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # Reduz velocidade
            else:
                speed_limit = MAX_SPEED  # Acelera se necessário

            throttle = 1.0 - (steering_angle_pred ** 2) - ((speed / speed_limit) ** 2)

            # Tempo de resposta do modelo
            response_time = time.time() - current_time

            # Cálculo do erro entre direção real e predita
            error = abs(steering_angle_real - steering_angle_pred)

            # Cálculo da distância percorrida
            time_diff = current_time - prev_time
            distance = (speed * 0.44704) * time_diff  # Convertendo mph para m/s e multiplicando pelo tempo
            total_distance += distance
            prev_time = current_time

            # Atualiza a média de velocidade
            speed_sum += speed
            speed_count += 1
            average_speed = speed_sum / speed_count if speed_count > 0 else 0

            # Detecção de colisão: Se a velocidade cair drasticamente, conta como colisão
            collision_flag = 0
            if prev_speed is not None and speed < prev_speed * 0.5:  # Redução brusca
                collision_flag = 1
                collision_count += 1

            prev_speed = speed
            prev_steering_angle = steering_angle_pred

            print(f"Predição -> Steering: {steering_angle_pred:.4f}, Throttle: {throttle:.4f}, Speed: {speed:.2f}, Error: {error:.4f}, Distance: {total_distance:.2f}m, Avg Speed: {average_speed:.2f}mph, Collisions: {collision_count}")

            # Salva métricas no CSV
            with open(metrics_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.utcnow(), steering_angle_pred, steering_angle_real, throttle, speed, error, response_time, collision_flag, total_distance, average_speed])

            send_control(steering_angle_pred, throttle)

        except Exception as e:
            print(f"Erro na predição: {e}")

@sio.on('connect')
def connect(sid, environ):
    print(f"Cliente conectado: {sid}")
    for _ in range(5):  # Teste por 5 iterações
        send_control(0.0, 0.5)  # Acelera por um curto período
        time.sleep(1)
    print("Teste de aceleração enviado.")

def send_control(steering_angle, throttle):
    """
    Envia os comandos para o simulador.
    """
    sio.emit(
        "steer",
        data={'steering_angle': str(steering_angle), 'throttle': str(throttle)},
        skip_sid=True
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Autonomous Driving")
    parser.add_argument("model", type=str, help="Caminho para o modelo treinado (ex: model.h5)")
    parser.add_argument("image_folder", type=str, nargs="?", default="", help="Pasta para salvar imagens capturadas durante a simulação.")
    args = parser.parse_args()

    # Carregar o modelo treinado
    print(f"Carregando modelo: {args.model}")
    model = load_model(args.model, custom_objects={'mse': MeanSquaredError()})

    # Criar a pasta para salvar imagens, se necessário
    if args.image_folder:
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)

    # Inicializando o servidor Flask + SocketIO
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(("0.0.0.0", 4567)), app)