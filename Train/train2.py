import pandas as pd
import numpy as np
import cv2
import os
import argparse
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau

np.random.seed(42)

def load_data(args):
    csv_path = os.path.join(args.data_dir, 'driving_log.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {csv_path}")

    data_df = pd.read_csv(csv_path, names=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])

    # Remover espaços extras dos caminhos dos arquivos
    for col in ['center', 'left', 'right']:
        data_df[col] = data_df[col].str.strip()

    # Ajustar os caminhos das imagens
    def fix_path(filename):
        return os.path.join(args.data_dir, "IMG", os.path.basename(filename))

    for col in ['center', 'left', 'right']:
        data_df[col] = data_df[col].apply(fix_path)

    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].astype(float).values

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

def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    correction = 0.2  

    while True:
        indices = np.random.permutation(len(image_paths))
        batch_images = []
        batch_steerings = []

        for i in range(batch_size):
            index = indices[i % len(image_paths)]
            img_choice = np.random.choice(['center', 'left', 'right'])

            if img_choice == 'center':
                img_path = image_paths[index][0]
                steering = steering_angles[index]
            elif img_choice == 'left':
                img_path = image_paths[index][1]
                steering = steering_angles[index] + correction
            else:
                img_path = image_paths[index][2]
                steering = steering_angles[index] - correction

            img = preprocess_image(img_path)

            if np.random.rand() < 0.5:  
                img = np.fliplr(img)
                steering = -steering

            batch_images.append(img)
            batch_steerings.append(steering)

        yield np.array(batch_images), np.array(batch_steerings)

def build_model():
    model = Sequential()
    model.add(Lambda(lambda x: x - 0.5, input_shape=(66, 200, 3)))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=0.0003, decay=1e-6)  
    model.compile(loss='mse', optimizer=optimizer)

    return model

def train_model(model, args, X_train, X_valid, y_train, y_valid):
    csv_logger = CSVLogger('training_log.csv', append=True)

    checkpoint = ModelCheckpoint(
        'model-{epoch:03d}.h5',
        monitor='val_loss',
        save_best_only=args.save_best_only,
        mode='auto',
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    model.fit(
        batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
        steps_per_epoch=max(1, args.samples_per_epoch // args.batch_size),
        epochs=args.nb_epoch,
        validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
        validation_steps=max(1, len(X_valid) // args.batch_size),
        callbacks=[checkpoint, early_stop, reduce_lr, csv_logger],
        verbose=1
    )

    model.save("model.h5")
    print("Modelo salvo como model.h5")

def main():
    parser = argparse.ArgumentParser(description='Treinamento de Rede Neural para Carro Autônomo')
    parser.add_argument('-d', dest='data_dir', type=str, default='data', help='Diretório dos dados')
    parser.add_argument('-t', dest='test_size', type=float, default=0.2, help='Proporção de dados para validação')
    parser.add_argument('-n', dest='nb_epoch', type=int, default=10, help='Número de épocas de treinamento')
    parser.add_argument('-s', dest='samples_per_epoch', type=int, default=20000, help='Amostras por época')
    parser.add_argument('-b', dest='batch_size', type=int, default=16, help='Tamanho do batch')
    parser.add_argument('-o', dest='save_best_only', type=bool, default=True, help='Salvar apenas melhores modelos')

    args = parser.parse_args()

    print("Carregando os dados...")
    X_train, X_valid, y_train, y_valid = load_data(args)

    print("Criando o modelo...")
    model = build_model()

    print("Iniciando treinamento...")
    train_model(model, args, X_train, X_valid, y_train, y_valid)

if __name__ == '__main__':
    main()