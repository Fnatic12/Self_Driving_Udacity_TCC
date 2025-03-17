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
from datetime import datetime
from flask import Flask
from io import BytesIO
from PIL import Image
from keras.models import load_model
from keras.losses import MeanSquaredError

# Inicializando o servidor SocketIO e Flask
sio = socketio.Server()
app = Flask(__name__)

# Inicializando o modelo e variáveis globais
model = None
prev_image_array = None

# Definição de velocidade máxima e mínima
MAX_SPEED = 25
MIN_SPEED = 10
speed_limit = MAX_SPEED

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
    if data:
        # Captura os dados enviados pelo simulador
        steering_angle = float(data["steering_angle"])
        throttle = float(data["throttle"])
        speed = float(data["speed"])

        # Decodifica a imagem da câmera
        img_string = data["image"]
        img_bytes = BytesIO(base64.b64decode(img_string))
        image = Image.open(img_bytes)

        try:
            # Processa a imagem antes de enviar para o modelo
            image = preprocess_image(image)
            image = np.array([image])  # Modelo espera um array 4D

            # Faz a predição do ângulo de direção
            steering_angle = float(model.predict(image, batch_size=1)[0][0])

            # Ajusta a aceleração conforme a velocidade do carro
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # Reduz velocidade
            else:
                speed_limit = MAX_SPEED  # Acelera se necessário

            throttle = 1.0 - (steering_angle ** 2) - ((speed / speed_limit) ** 2)

            print(f"🔄 Predição -> Steering: {steering_angle:.4f}, Throttle: {throttle:.4f}, Speed: {speed:.2f}")
            send_control(steering_angle, throttle)
        
        except Exception as e:
            print(f"⚠️ Erro na predição: {e}")

        # Salvar imagens se a opção foi ativada
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save(f"{image_filename}.jpg")

    else:
        sio.emit('manual', data={}, skip_sid=True)

@sio.on('connect')
def connect(sid, environ):
    print(f"✅ Cliente conectado: {sid}")
    for _ in range(5):  # Teste por 5 iterações
        send_control(0.0, 0.5)  # Acelera por um curto período
        time.sleep(1)
    print("🚗 Teste de aceleração enviado.")

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

    # Carregar o modelo treinado com suporte a custom_objects
    print(f"🔄 Carregando modelo: {args.model}")
    model = load_model(args.model, custom_objects={'mse': MeanSquaredError()})

    # Criar a pasta para salvar imagens, se necessário
    if args.image_folder:
        print(f"📁 Criando pasta de imagens em {args.image_folder}")
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("📸 Salvando imagens da simulação...")

    # Inicializando o servidor Flask + SocketIO
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(("0.0.0.0", 4567)), app)