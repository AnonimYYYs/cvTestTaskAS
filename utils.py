import numpy as np
import matplotlib.pyplot as plt

from figures import Drawer


batch_size = 32
epochs = 10
epoch_size = 1000
test_size = 100

# словарь для простой категоризации
name_set = {
    "Empty": 0,
    "Rhombus": 1,
    "Triangle": 2,
    "Circle": 3,
    "Hexagon": 4,
}

# для генерации датасета "на лету"
def generate_set(size, add_hexagon=True):
    d = Drawer()
    X_array = []  # Массив для хранения изображений
    y_array = []  # Массив для хранения данных о фигурах

    y_objects_count = [0, 0, 0, 0, 0]  # Массив для хранения количества объектов на изображении
    y_objects_types = [0, 0, 0, 0]  # Массив для хранения типов объектов на изображении


    for i in range(size):
        img, desc = d.generate_image(add_hexagon=add_hexagon)

        # Сохраняем изображение и его аугментации в массив X
        X_array.append(img)

        # Создаем массив для хранения данных о фигурах
        shapes_data = []

        y_objects_count[len(desc) - 1] += 1


        # Обрабатываем словарь с данными о фигурах
        for j in range(5):
            if j < len(desc):
                shape = desc[j]
                name = shape["name"]
                x = shape["region"]["origin"]["x"]
                y = shape["region"]["origin"]["y"]
                width = shape["region"]["size"]["width"]
                height = shape["region"]["size"]["height"]
                y_objects_types[name_set[name] - 1] += 1
            else:
                # Если в словаре меньше 5 объектов, добавляем пустые данные
                name = "Empty"
                x = y = width = height = 0

            # Добавляем данные о фигуре в массив
            for value in [x, y, width, height]:
                shapes_data.append(value)

        # Добавляем массив shapes_data в массив y
        y_array.append(shapes_data)

    X_array = np.array([np.array(x) for x in X_array])
    y_array = np.array(y_array)

    return X_array, y_array, y_objects_count, y_objects_types


# Рассчитаем IoU, Precision и Recall
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    intersection_x = max(x1, x2)
    intersection_y = max(y1, y2)
    intersection_w = min(x1 + w1, x2 + w2) - intersection_x
    intersection_h = min(y1 + h1, y2 + h2) - intersection_y

    if intersection_w <= 0 or intersection_h <= 0:
        return 0.0

    intersection_area = intersection_w * intersection_h
    union_area = (w1 * h1) + (w2 * h2) - intersection_area

    return intersection_area / union_area


def test_model(model, X_test, y_test):

    # Оценим модель на тестовых данных
    y_pred = model.predict(X_test, verbose=0)

    # Разбиваем предсказанные параметры на отдельные прямоугольники
    y_pred = y_pred.reshape(-1, 5, 4)
    y_test = y_test.reshape(-1, 5, 4)

    # Рассчитываем IoU, Precision и Recall для каждого изображения
    iou_scores = []
    precisions = []
    recalls = []

    for true_boxes, pred_boxes in zip(y_test, y_pred):
        iou_scores_per_image = []
        precision_per_image = []
        recall_per_image = []

        for true_box, pred_box in zip(true_boxes, pred_boxes):
            iou = calculate_iou(true_box, pred_box)
            iou_scores_per_image.append(iou)

            if iou > 0.5:
                precision_per_image.append(1)
            else:
                precision_per_image.append(0)

            recall_per_image.append(1 if iou > 0.5 else 0)

        iou_scores.append(iou_scores_per_image)
        precisions.append(precision_per_image)
        recalls.append(recall_per_image)

    iou_scores = np.array(iou_scores)
    precisions = np.array(precisions)
    recalls = np.array(recalls)

    # Рассчитываем максимальное, минимальное и среднее IoU
    max_iou = np.max(iou_scores)
    min_iou = np.min(iou_scores)
    average_iou = np.mean(iou_scores)

    # Рассчитываем Precision и Recall для IoU > 0.5
    precision = np.sum(precisions) / np.sum(precisions + (1 - precisions))
    recall = np.sum(recalls) / np.sum(recalls + (1 - recalls))

    print(f"Максимальное IoU:\t{max_iou}")
    print(f"Минимальное IoU:\t{min_iou}")
    print(f"Среднее IoU:\t\t{average_iou}")
    print(f"Precision:\t\t\t{precision}")
    print(f"Recall:\t\t\t\t{recall}")
