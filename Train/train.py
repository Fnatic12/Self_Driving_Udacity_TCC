import pandas as pd
import numpy as np
import cv2
import os
import argparse
import datetime
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, Dropout, Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau

##python train3.py -d ./data -c driving_log.csv -t 0.2 -n 100 -b 16 -o True -m nvidia##
#python train3.py -d ./data -c driving_log.csv -t 0.2 -n 100 -b 16 -o True -m comma#

np.random.seed(42)

def load_data(args):
    csv_path = os.path.join(args.data_dir, args.csv_file)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {csv_path}")

    data_df = pd.read_csv(csv_path, names=['img_center', 'img_left', 'img_right', 'ang_right', 'ang_left', 'throttle', 'speed'])

    data_df['img_center'] = data_df['img_center'].apply(lambda x: os.path.join(args.data_dir, "IMG", os.path.basename(str(x).strip())))
    data_df['steering'] = (data_df['ang_left'].astype(float) + data_df['ang_right'].astype(float)) / 2
    data_df = data_df[data_df['img_center'].apply(os.path.exists)]

    X = data_df['img_center'].values
    y = data_df[['steering', 'throttle']].astype(float).values

    return train_test_split(X, y, test_size=args.test_size, random_state=42)

def preprocess_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Imagem não encontrada: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Erro ao carregar imagem: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 66))
    return img / 255.0

def batch_generator(image_paths, targets, batch_size, is_training):
    while True:
        indices = np.random.permutation(len(image_paths))
        batch_images = []
        batch_targets = []

        for i in range(batch_size):
            index = indices[i % len(image_paths)]
            img_path = image_paths[index]
            steering, throttle = targets[index]

            img = preprocess_image(img_path)

            if is_training and np.random.rand() < 0.5:
                img = np.fliplr(img)
                steering = -steering

            batch_images.append(img)
            batch_targets.append([steering, throttle])

        batch_targets = np.array(batch_targets)
        yield np.array(batch_images), {
            'steering': batch_targets[:, 0],
            'throttle': batch_targets[:, 1]
        }

def build_model_nvidia():
    inputs = Input(shape=(66, 200, 3))
    x = Lambda(lambda x: x - 0.5)(inputs)
    x = Conv2D(24, (5, 5), strides=(2, 2), activation='elu')(x)
    x = Conv2D(36, (5, 5), strides=(2, 2), activation='elu')(x)
    x = Conv2D(48, (5, 5), strides=(2, 2), activation='elu')(x)
    x = Conv2D(64, (3, 3), activation='elu')(x)
    x = Conv2D(64, (3, 3), activation='elu')(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(100, activation='elu')(x)
    x = Dense(50, activation='elu')(x)
    x = Dense(10, activation='elu')(x)

    steering_output = Dense(1, name='steering')(x)
    throttle_output = Dense(1, name='throttle')(x)

    model = Model(inputs=inputs, outputs=[steering_output, throttle_output])

    model.compile(
        optimizer=Adam(learning_rate=0.0003, decay=1e-6),
        loss={'steering': 'mse', 'throttle': 'mse'},
        loss_weights={'steering': 1.0, 'throttle': 1.0}
    )

    return model

def build_model_comma():
    inputs = Input(shape=(66, 200, 3))
    x = Lambda(lambda x: x / 255.0 - 0.5)(inputs)
    x = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(x)
    x = Conv2D(32, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Conv2D(64, (5, 5), strides=(2, 2), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)

    steering_output = Dense(1, name='steering')(x)
    throttle_output = Dense(1, name='throttle')(x)

    model = Model(inputs=inputs, outputs=[steering_output, throttle_output])

    model.compile(
        optimizer=Adam(learning_rate=0.0003),
        loss={'steering': 'mse', 'throttle': 'mse'},
        loss_weights={'steering': 1.0, 'throttle': 1.0}
    )

    return model

def train_model(model, args, X_train, X_valid, y_train, y_valid):
    log_name = f"training_log_{args.model_name}.csv"
    csv_logger = CSVLogger(log_name, append=True)
    checkpoint = ModelCheckpoint(f"model_{args.model_name}-{{epoch:03d}}.h5", monitor='val_loss', save_best_only=args.save_best_only, mode='auto', verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    model.fit(
        batch_generator(X_train, y_train, args.batch_size, True),
        steps_per_epoch=max(1, len(X_train) // args.batch_size),
        epochs=args.nb_epoch,
        validation_data=batch_generator(X_valid, y_valid, args.batch_size, False),
        validation_steps=max(1, len(X_valid) // args.batch_size),
        callbacks=[checkpoint, early_stop, reduce_lr, csv_logger],
        verbose=1
    )

    model_name = f"model_{args.model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.h5"
    model.save(model_name)
    print(f"✅ Modelo salvo como {model_name}")

def main():
    parser = argparse.ArgumentParser(description='Treinamento de Rede Neural para Carro Autônomo')
    parser.add_argument('-d', dest='data_dir', type=str, required=True, help='Diretório dos dados')
    parser.add_argument('-c', dest='csv_file', type=str, default='driving_log.csv', help='Nome do CSV')
    parser.add_argument('-t', dest='test_size', type=float, default=0.2, help='Proporção de validação')
    parser.add_argument('-n', dest='nb_epoch', type=int, default=10, help='Épocas')
    parser.add_argument('-b', dest='batch_size', type=int, default=16, help='Tamanho do batch')
    parser.add_argument('-o', dest='save_best_only', type=bool, default=True, help='Salvar melhores modelos')
    parser.add_argument('-m', dest='model_name', type=str, required=True, choices=['nvidia', 'comma'], help='Nome do modelo para treinar')

    args = parser.parse_args()

    print("📥 Carregando os dados...")
    X_train, X_valid, y_train, y_valid = load_data(args)

    print("🧠 Criando o modelo...")
    model = build_model_nvidia() if args.model_name == 'nvidia' else build_model_comma()

    print("🚗 Iniciando treinamento...")
    train_model(model, args, X_train, X_valid, y_train, y_valid)

if __name__ == '__main__':
    main()
