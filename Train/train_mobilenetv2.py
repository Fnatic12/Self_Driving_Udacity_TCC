import pandas as pd
import numpy as np
import cv2
import os
import argparse
import datetime
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.applications import MobileNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau, Callback

np.random.seed(42)

def load_data(args):
    csv_path = os.path.join(args.data_dir, args.csv_file)
    data_df = pd.read_csv(csv_path, sep=';')
    data_df = data_df[data_df['img_center'].apply(os.path.exists)]
    data_df['steering'] = (data_df['ang_left'].astype(float) + data_df['ang_right'].astype(float)) / 2
    X = data_df['img_center'].values
    y = data_df['steering'].values
    return train_test_split(X, y, test_size=args.test_size, random_state=42)

def preprocess_image_tf(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, label

def load_dataset(X, y, batch_size, is_training):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if is_training:
        dataset = dataset.shuffle(buffer_size=2048)
    dataset = dataset.map(preprocess_image_tf, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def build_model():
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # congelar o backbone
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(50, activation='relu')(x)
    predictions = Dense(1)(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0003), loss='mse')
    return model

class EpochTimer(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.start = time.time()
    def on_epoch_end(self, epoch, logs=None):
        print(f"⏱️ Tempo da epoch {epoch+1}: {time.time() - self.start:.2f}s")

def train_model(model, args, train_ds, valid_ds):
    csv_logger = CSVLogger('training_log_mobilenet.csv', append=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=2)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=2)
    timer = EpochTimer()

    model.fit(
        train_ds,
        epochs=args.nb_epoch,
        validation_data=valid_ds,
        callbacks=[early_stop, reduce_lr, csv_logger, timer],
        verbose=2
    )
    model_name = f"mobilenet_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.h5"
    model.save(model_name)
    print(f"✅ Modelo salvo como {model_name}")

def main():
    parser = argparse.ArgumentParser(description='Treinamento com MobileNetV2 para Carro Autônomo')
    parser.add_argument('-d', dest='data_dir', type=str, required=True)
    parser.add_argument('-c', dest='csv_file', type=str, default='driving_log_cleaned.csv')
    parser.add_argument('-t', dest='test_size', type=float, default=0.2)
    parser.add_argument('-n', dest='nb_epoch', type=int, default=100)
    parser.add_argument('-b', dest='batch_size', type=int, default=32)
    parser.add_argument('-o', dest='save_best_only', type=bool, default=True)

    args = parser.parse_args()

    print("📅 Carregando os dados...")
    X_train, X_valid, y_train, y_valid = load_data(args)

    print("📦 Preparando datasets...")
    train_ds = load_dataset(X_train, y_train, args.batch_size, True)
    valid_ds = load_dataset(X_valid, y_valid, args.batch_size, False)

    print("🧠 Criando o modelo MobileNetV2...")
    model = build_model()

    print("🚗 Iniciando treinamento...")
    train_model(model, args, train_ds, valid_ds)

if __name__ == '__main__':
    main()
