from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py


images_per_class = 40           # Количество изображений в датасете
fixed_size = tuple((500, 500))  # Фиксированный размер изображения
train_path = r'data\train'      # Путь до тренировочного датасета
h5_data = r'data\data.h5'       # Путь до данных обучения
h5_labels = r'data\labels.h5'   # Путь до названий классов
bins = 8                        # Количество бинов для гистограммы


def fd_hu_moments(image):
    """Фича формы. Вычисление моментов для анализа формы"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


def fd_haralick(image):
    """Фича текстуры. Определение текстуры, используя Haralick Textures"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick


def fd_histogram(image):
    """Фича цвета. Вычисление цветовой гистограммы изображения"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


if __name__ == "__main__":
    # Получаем названия покемонов и сортировка
    train_labels = os.listdir(train_path)
    train_labels.sort()

    # Список фич для каждого изображения из датасета
    global_features = []
    labels = []

    # Тренировка на датасете
    for name in train_labels:
        # Путь до папки с датасетом конкретного класса
        dir = os.path.join(train_path, name)
        current_label = name

        # Цикл тренировки для данного класса на датасете
        for x in range(1, images_per_class+1):
            # Название файла
            file = dir + '\\' + str(x) + '.jpg'
            # Чтение изображение и изменение размера
            image = cv2.imread(file)
            image = cv2.resize(image, fixed_size)
            # Вычисление фич
            fv_hu_moments = fd_hu_moments(image)
            fv_haralick = fd_haralick(image)
            fv_histogram = fd_histogram(image)
            # Объединение фич в один массив
            global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
            # Помещаем вычисленный массив в список всех фич
            global_features.append(global_feature)
            # Добавляем название целевого класса, которому соответствует фича
            labels.append(current_label)

    # Кодирует названия покемонов числами от 0 до 4
    # и возвращает список
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(labels)

    # Нормализация значений в интервал от 0 до 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(global_features)

    # Сохраняем список фич в файл с расширением .h5
    h5f_data = h5py.File(h5_data, 'w')
    h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))
    # Сохраняем список меток класса в файл с расширением .h5
    h5f_label = h5py.File(h5_labels, 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(target))
    h5f_data.close()
    h5f_label.close()
