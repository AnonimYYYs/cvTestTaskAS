import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import MeanIoU, Precision, Recall

import matplotlib.pyplot as plt

from utils import *

epochs = 3

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

    X_train_12000, y_train_12000, _, _ = generate_set(12000, add_hexagon=False)
    X_test_3000, y_test_3000, _, _ = generate_set(3000, add_hexagon=False)
    X_test_3000_2, y_test_3000_2, _, _ = generate_set(3000)

    # Обучаем модель на 12000 изображениях
    history = model.fit(X_train_12000, y_train_12000, batch_size=batch_size, epochs=5)

    # проверяем на изображениях без гексагона
    print("\n3000 без гексагона")
    test_model(model, X_test_3000, y_test_3000)

    # проверяем на изображениях без гексагона
    print("\n3000 с гексагоном")
    test_model(model, X_test_3000_2, y_test_3000_2)

    model.save('start_learning.keras')

    for i in range(400, 8001, 400):
        print(f"\n\n\tДополнительно изображений: {i}\n")

        X_train_new, y_train_new, _, _ = generate_set(i)

        X_train = np.concatenate((X_train_12000, X_train_new), axis=0)
        y_train = np.concatenate((y_train_12000, y_train_new), axis=0)

        model_copy = keras.models.load_model('start_learning.keras')

        # Дополнительно обучаем модель на 12000+i изображениях
        history = model_copy.fit(X_train_12000, y_train_12000, batch_size=batch_size, epochs=epochs, verbose=2)

        # проверяем на изображениях без гексагона
        print("3000 без гексагона")
        test_model(model_copy, X_test_3000, y_test_3000)

        # проверяем на изображениях без гексагона
        print("3000 с гексагоном")
        test_model(model_copy, X_test_3000_2, y_test_3000_2)





