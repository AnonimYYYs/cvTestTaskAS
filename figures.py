import random
import math
from PIL import Image, ImageDraw


def random_color():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)


class Figure:
    def __init__(self, identifier, params):
        # определяем параметры
        self._id = identifier
        self._params = params
        self._points = dict()
        # генерируем
        self.create()
        # не забываем скорректировать x, y, w, h, так как в процессе генерации они могут сместиться
        self.fix_params()

    def isIntersects(self, other):
        return (
            self._params['x'] < other._params['x'] + other._params['w'] and
            self._params['x'] + self._params['w'] > other._params['x'] and
            self._params['y'] < other._params['y'] + other._params['h'] and
            self._params['y'] + self._params['h'] > other._params['y']
        )

    def create(self):
        raise Exception('Not implemented')

    def draw(self, canvas, color):
        canvas.polygon([tuple(i) for i in self._points], fill=color, outline=color)

    def fix_params(self):
        self._params['x'] = min([point[0] for point in self._points])
        self._params['y'] = min([point[1] for point in self._points])
        self._params['w'] = max([point[0] for point in self._points]) - self._params['x']
        self._params['h'] = max([point[1] for point in self._points]) - self._params['y']

    def describe(self):
        return {
            "id": self._id,
            "name": type(self).__name__,
            "region": {
                "origin": {
                    "x": self._params['x'],
                    "y": self._params['y']
                },
                "size": {
                    "width": self._params['w'],
                    "height": self._params['h']
                }
            },
        }


class Rhombus(Figure):
    def create(self):
        # # создаем шаблон фигуры
        x, y, w, h = self._params["x"], self._params["y"], self._params["w"], self._params["h"]
        angle = random.randint(0, 360)

        # создаем ромб по заданным характеристикам
        points = [(x + w // 2, y), (x + w, y + h // 2), (x + w // 2, y + h), (x, y + h // 2)]

        # поворачиваем ромб
        points = [(x + int((px - x) * math.cos(math.radians(angle)) - (py - y) * math.sin(math.radians(angle))),
                   y + int((px - x) * math.sin(math.radians(angle)) + (py - y) * math.cos(math.radians(angle))))
                  for px, py in points]

        self._points = points


class Triangle(Figure):
    def create(self):
        # создаем шаблон фигуры
        x, y, w, h = self._params["x"], self._params["y"], self._params["w"], self._params["h"]
        angle = random.randint(0, 360)
        points = [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(3)]

        # поворачиваем
        points = [
            [
                x + (px - x) * math.cos(math.radians(angle)) - (py - y) * math.sin(math.radians(angle)),
                y + (px - x) * math.sin(math.radians(angle)) + (py - y) * math.cos(math.radians(angle))
            ]
            for px, py in points
        ]

        # нормализуем по размеру
        min_x = min(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_x = max(point[0] for point in points)
        max_y = max(point[1] for point in points)

        for i in range(len(points)):
            points[i][0] = int(x + (points[i][0] - min_x) / (max_x - min_x) * w)
            points[i][1] = int(y + (points[i][1] - min_y) / (max_y - min_y) * h)

        self._points = points


class Circle(Figure):
    """
    self._points = [[x, y], r]; (x,y) - центр круга, r - радиус

    данному классу необходимы некоторые отдельные методы, так как это круг, не многоугольник
    """
    def create(self):
        # создаем шаблон фигуры
        x, y, w, h = self._params["x"], self._params["y"], self._params["w"], self._params["h"]

        # вычисляем центр и радиус
        cx = int(x + w / 2)
        cy = int(y + h / 2)

        r = int(min([w, h]) / 2)

        self._points = [[cx, cy], r]

    def draw(self, canvas, color):
        canvas.ellipse([
            (self._points[0][0] - self._points[1], self._points[0][1] - self._points[1]),
            (self._points[0][0] + self._points[1], self._points[0][1] + self._points[1])
        ], fill=color, outline=color)

    def fix_params(self):
        self._params['x'] = self._points[0][0] - self._points[1]
        self._params['y'] = self._points[0][1] - self._points[1]
        self._params['w'] = self._points[1] * 2
        self._params['h'] = self._points[1] * 2


class Hexagon(Figure):
    def create(self):
        # создаем шаблон фигуры
        x, y, w, h = self._params["x"], self._params["y"], self._params["w"], self._params["h"]
        angle = random.randint(0, 360)
        angle_rad = math.radians(angle)
        points = [
            [1.0, 0.0],
            [0.5, 0.87],
            [-0.5, 0.87],
            [-1.0, 0.0],
            [-0.5, -0.87],
            [0.5, -0.87],
        ]

        # поворачиваем с учетом центра в (0, 0)
        points = [
            [
                (xp * math.cos(angle_rad) - yp * math.sin(angle_rad)),
                (xp * math.sin(angle_rad) + yp * math.cos(angle_rad))
            ]
            for xp, yp in points
        ]

        # расширяем до размеров w<->h, так чтобы не нарушить "правильность"
        k = min(
            w / 2 * max([point[0] for point in points]),
            h / 2 * max([point[1] for point in points])
        )
        # добавляем параллельный перенос до (x, y)
        points = [[int(point[0] * k + x), int(point[1] * k + y)] for point in points]



        self._points = points


class Drawer:
    def __init__(self):
        # # набор метаданных генератора
        # self._image_size = (256, 256)
        # self._tries = 5
        # self._min_shape = 25
        # self._max_shape = 150
        # self._index = 0
        # self._max_create_figures = 40
        # self._max_figures = 5
        # self._dropout = 0.0
        # набор метаданных генератора
        self._image_size = (256, 256)
        self._tries = 5
        self._min_shape = 25
        self._max_shape = 150
        self._index = 0
        self._max_create_figures = 10
        self._max_figures = 5
        self._dropout = 0.75


    def generate_image(self, add_hexagon=True):
        self._index = 0
        image = Image.new("RGB", self._image_size, random_color())
        draw = ImageDraw.Draw(image)

        figures = []

        # генерируем пока есть необходимость или пока не достигли предела
        while len(figures) < self._max_figures and self._index < self._max_create_figures:

            # исключаем фигуру с определенным шансом (иначе на каждом изображении будет по 5 штук в 99%
            if (random.random() <= self._dropout) and (len(figures) != 0):
                self._index += 1
                continue

            # генерируем фигуру
            x = random.randint(0, self._image_size[0] - self._min_shape)
            y = random.randint(0, self._image_size[1] - self._min_shape)

            w = random.randint(self._min_shape, min(self._max_shape, self._image_size[0] - x))
            h = random.randint(self._min_shape, min(self._max_shape, self._image_size[1] - y))

            figs = [Rhombus, Triangle, Circle]
            if add_hexagon:
                figs.append(Hexagon)

            figureType = random.choice(figs)
            figure = figureType(self._index, {'x': x, 'y': y, 'w': w, 'h': h})

            is_valid = True

            # смотрим, пересекается ли она с уже существующими
            for other_figure in figures:
                if figure.isIntersects(other_figure):
                    is_valid = False
                    break

            # если нет - добавляем как новую
            if is_valid:
                figure.draw(draw, random_color())
                figures.append(figure)
                self._index += 1

        # не забываем про словарь для .json
        describes = [fig.describe() for fig in figures]

        return image, describes


