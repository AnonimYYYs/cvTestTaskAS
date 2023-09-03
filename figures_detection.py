import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import MeanIoU, Precision, Recall

import matplotlib.pyplot as plt

from utils import *


if __name__ == "__main__":

    # Создаем модель CNN
    model = keras.Sequential([
        layers.Input(shape=(256, 256, 3)),  # Размер изображений
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(20, activation='relu'),
    ])

    # Компилируем модель с выбранной функцией потерь и оптимизатором
    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )

    X_train, y_train, stat_count, stat_shape = generate_set(epoch_size)
    X_test, y_test, _, _ = generate_set(test_size)

    print('Количество различных видов фигур:')
    for i in range(4):
        print(f'\t {list(name_set.keys())[i + 1]}: {stat_shape[i]}')
    print('Количество фигур на изображении:')
    for i in range(5):
        print(f'\t {i+1}: {stat_count[i]}')

    # Обучаем модель
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    test_model(model, X_test, y_test)
