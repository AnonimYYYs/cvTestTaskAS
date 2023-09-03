import os
import json

from figures import Drawer

images_amount = 100
images_folder = 'images'

# Создаем папку 'images', если ее нет
if not os.path.exists(images_folder):
    os.mkdir(images_folder)

d = Drawer()

for i in range(images_amount):

    img, desc = d.generate_image()

    image_filename = f'{images_folder}/image_{i:03d}.png'
    img.save(image_filename, 'PNG', compress_level=0)

    # Преобразуем словарь в JSON и сохраняем в файл
    description_filename = f'{images_folder}/{i:03d}.json'  # Создаем имя файла в формате "001.json"
    with open(description_filename, 'w') as file:
        json.dump(desc, file, indent=4, ensure_ascii=False)