import numpy as np
import h5py
import cv2
import os


FIXED_SIZE = tuple((28, 28))                                # Фиксированный размер изображения
TRAIN_PATH = r'data\train'                                  # Путь до тренировочного датасета
TEST_PATH = r'data\test'                                    # Путь до тестового датасета
H5_DATA = r'data\output\data.h5'                            # Путь до данных обучения
H5_LABELS = r'data\output\labels.h5'                        # Путь до названий классов
NOISER_MODEL_PATH = r'data\output\noiser_model'             # Путь до модели НС добавления шума
AUTOENCODER_MODEL_PATH = r'data\output\autoencoder_model'   # Путь до модели НС автокодировщика


if __name__ == "__main__":
    # Получаем название классов по имени каталога
    train_labels = os.listdir(TRAIN_PATH)

    # Списки для хранения данных обучения и соответствующих
    # им меток - названий классов
    data = []
    labels = []

    # Цикл по каталогу с классами
    for class_name in train_labels:
        # Путь до папки с датасетом конкретного класса
        dir = os.path.join(TRAIN_PATH, class_name)
        current_label = class_name

        # Цикл по изображениям в папке класса
        for img in os.scandir(dir):
            # Чтение изображение и изменение размера
            image = cv2.imread(img.path, 0)
            image = cv2.resize(image, FIXED_SIZE)
            # Приведение значений к интервалу [0, 1] и сглаживание для FFNN
            #image_FFNN = image.flatten() / 255.0
            image_CNN = image / 255.0
            # Сохраняем данные и соответствующее им название класса в списках
            labels.append(int(current_label))
            #data.append(image_FFNN)
            data.append(image_CNN)

    h5f_data = h5py.File(H5_DATA, 'w')
    h5f_data.create_dataset('dataset_1', data=np.array(data))
    h5f_data.close()
    h5f_label = h5py.File(H5_LABELS, 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(labels))
    h5f_label.close()
