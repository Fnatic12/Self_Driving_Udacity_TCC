import pandas as pd
import numpy as np
import cv2
import os
import argparse
import datetime
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau

np.random.seed(42)

def load_data(args):
    csv_path = os.path.join(args.data_dir, args.csv_file)
    data_df = pd.read_csv(csv_path, sep=';')
    data_df = data_df[data_df['img_center'].apply(os.path.exists)]
    data_df['steering'] = (data_df['ang_left'].astype(float) + data_df['ang_right'].astype(float)) / 2
    X = data_df['img_center'].values
    y = data_df['steering'].values
    return train_test_split(X, y, test_size=args.test_size, random_state=42)

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 66))
    return img / 255.0

def batch_generator(image_paths, steering_angles, batch_size, is_training):
    while True:
        indices = np.random.permutation(len(image_paths))
        batch_images, batch_steerings = [], []
        for i in range(batch_size):
            index = indices[i % len(image_paths)]
            img = preprocess_image(image_paths[index])
            steering = steering_angles[index]
            if is_training and np.random.rand() < 0.5:
                img = np.fliplr(img)
                steering = -steering
            batch_images.append(img)
            batch_steerings.append(steering)
        yield np.array(batch_images), np.array(batch_steerings)

def build_comma_ai():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))
    model.add(Conv2D(16, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(32, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.0003), loss='mse')
    return model

def train_model(model, args, X_train, X_valid, y_train, y_valid):
    csv_logger = CSVLogger('training_log_commaai.csv', append=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    model.fit(
        batch_generator(X_train, y_train, args.batch_size, True),
        steps_per_epoch=len(X_train) // args.batch_size,
        epochs=args.nb_epoch,
        validation_data=batch_generator(X_valid, y_valid, args.batch_size, False),
        validation_steps=len(X_valid) // args.batch_size,
        callbacks=[early_stop, reduce_lr, csv_logger],
        verbose=1
    )
    model_name = f"model_commaai_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.h5"
    model.save(model_name)
    print(f"✅ Modelo salvo como {model_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='data_dir', type=str, required=True)
    parser.add_argument('-c', dest='csv_file', type=str, default='driving_log_cleaned.csv')
    parser.add_argument('-t', dest='test_size', type=float, default=0.2)
    parser.add_argument('-n', dest='nb_epoch', type=int, default=10)
    parser.add_argument('-b', dest='batch_size', type=int, default=8)
    args = parser.parse_args()

    print("📥 Carregando os dados...")
    X_train, X_valid, y_train, y_valid = load_data(args)
    print("🧠 Criando o modelo Comma.ai...")
    model = build_comma_ai()
    print("🚗 Iniciando treinamento...")
    train_model(model, args, X_train, X_valid, y_train, y_valid)

if __name__ == '__main__':
    main()