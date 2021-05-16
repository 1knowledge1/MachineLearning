import numpy as np
import h5py
import cv2
import os
from sklearn.preprocessing import LabelBinarizer


FIXED_SIZE = tuple((32, 32))                   # Фиксированный размер изображения
TRAIN_PATH = r'data\train'                     # Путь до тренировочного датасета
TEST_PATH = r'data\test'                       # Путь до тестового датасета
H5_DATA_FFNN = r'data\output\data_FFNN.h5'     # Путь до данных обучения для FFNN
H5_DATA_CNN = r'data\output\data_CNN.h5'       # Путь до данных обучения для CNN
H5_LABELS = r'data\output\labels.h5'           # Путь до названий классов
MODEL_FFNN_PATH = 'data\output\FFNN_model'     # Путь до модели НС прямого распространения
MODEL_CNN_PATH = 'data\output\CNN_model'       # Путь до модели сверточной НС
SEED = 17                                      # Значение для воспроизводимости результатов
TEST_SIZE = 0.15                               # Процент данных для тестовой выборки


if __name__ == "__main__":
    # Получаем название классов по имени каталога
    train_labels = os.listdir(TRAIN_PATH)

    # Списки для хранения данных обучения и соответствующих
    # им меток - названий классов
    data_FFNN = []
    data_CNN = []
    labels = []

    # Цикл по каталогу с классами
    for class_name in train_labels:
        # Путь до папки с датасетом конкретного класса
        dir = os.path.join(TRAIN_PATH, class_name)
        current_label = class_name

        # Цикл по изображениям в папке класса
        for img in os.scandir(dir):
            # Чтение изображение и изменение размера
            image = cv2.imread(img.path)
            image = cv2.resize(image, FIXED_SIZE)
            # Приведение значений к интервалу [0, 1] и сглаживание для FFNN
            image_FFNN = image.flatten() / 255.0
            image_CNN = image / 255.0
            # Сохраняем данные и соответствующее им название класса в списках
            labels.append(current_label)
            data_FFNN.append(image_FFNN)
            data_CNN.append(image_CNN)

    # Конвертируем метки из целых чисел в векторы с помощью LabelBinarizer
    lb = LabelBinarizer()
    labels_transform = lb.fit_transform(labels)

    # Сохраняем списки данных и меток в файлы с расширением .h5
    h5f_data_FFNN = h5py.File(H5_DATA_FFNN, 'w')
    h5f_data_FFNN.create_dataset('dataset_1', data=np.array(data_FFNN))
    h5f_data_FFNN.close()
    h5f_data_CNN = h5py.File(H5_DATA_CNN, 'w')
    h5f_data_CNN.create_dataset('dataset_1', data=np.array(data_CNN))
    h5f_data_CNN.close()
    h5f_label = h5py.File(H5_LABELS, 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(labels_transform))
    h5f_label.close()
