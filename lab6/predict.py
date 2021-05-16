import warnings
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from prepare_sample import *


def get_data():
    """Получить данные из файлов .h5

    Возвращает:
    data_FFNN (list): список данных для FFNN
    data_CNN (list): список данных для CNN
    labels (list): список меток

    """
    h5f_data_FFNN = h5py.File(H5_DATA_FFNN, 'r')
    h5f_data_CNN = h5py.File(H5_DATA_CNN, 'r')
    h5f_label = h5py.File(H5_LABELS, 'r')
    data_FFNN = np.array(h5f_data_FFNN['dataset_1'])
    data_CNN = np.array(h5f_data_CNN['dataset_1'])
    labels = np.array(h5f_label['dataset_1'])
    h5f_data_FFNN.close()
    h5f_data_CNN.close()
    h5f_label.close()
    return data_FFNN, data_CNN, labels


def estimate_model(model, test_data, test_labels, class_names, history):
    """Оценка модели с выводом графика потерь и точности

    Параметры:
    model (): модель НС
    test_data (list): список тестовых данных
    test_labels (list): список меток соответствующих тестовым данным
    history ():

    """
    # Оценка модели с помощью .predict и classification_report
    predictions = model.predict(test_data, batch_size=32)
    print(classification_report(test_labels.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=class_names))

    # Строим графики потерь и точности
    N = history.epoch
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, history.history["loss"], label="train_loss")
    plt.plot(N, history.history["val_loss"], label="val_loss")
    plt.plot(N, history.history["accuracy"], label="train_acc")
    plt.plot(N, history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


def create_model_FFNN(data, labels, class_names, test_size, seed, model_path):
    """Создать и обучить модель FFNN, сохранить по пути model_path

    Параметры:
    data (list): список даных для тренировки НС
    labels (list): список меток соответствующих данным
    class_names (list): названия классов
    test_size (float): процент данных для тестирования [0, 1]
    seed (int): параметр для ГСЧ
    model_path (list): путь, куда сохранять модель

    """
    # Разделение выборки на тренировочную и тестовую
    (train_data, test_data, train_labels, test_labels) = train_test_split(np.array(data),
                                                                          np.array(labels),
                                                                          test_size=test_size,
                                                                          random_state=seed)
    # Определение архитектуры НС прямого распространения 3072-2048-1024-3
    # с исключением нейронов (Dropout) для предотвращения переобучения
    # ф-ия активации - relu: F(x) = max(0,x)
    model = Sequential()
    #model.add(Dense(2048, input_dim=3072, activation='relu', kernel_initializer='uniform'))
    model.add(Dense(2048, input_dim=3072, activation='relu'))
    model.add(Dropout(0.1))
    #model.add(Dense(1024, activation='relu', kernel_initializer='uniform'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='softmax'))

    # Компилируем модель, используя SGD как оптимизатор и категориальную
    # кросс-энтропию в качестве функции потерь
    print("[INFO] Компиляция модели...")
    optimizer = SGD(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model.summary()

    # Обучаем модель, epochs - количество эпох, batch_size - контролирует размер пакета данных для передачи по сети
    print("[INFO] Обучение модели...")
    history = model.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=28, batch_size=32)
    print("[INFO] Оценка модели...")
    estimate_model(model, test_data, test_labels, class_names, history)
    print("[INFO] Сериализация модели...")
    model.save(model_path)


def create_model_CNN(data, labels, class_names, test_size, seed, model_path):
    """Создать и обучить модель CNN, сохранить по пути model_path

    Параметры:
    data (list): список даных для тренировки НС
    labels (list): список меток соответствующих данным
    class_names (list): названия классов
    test_size (float): процент данных для тестирования [0, 1]
    seed (int): параметр для ГСЧ
    model_path (list): путь, куда сохранять модель

    """
    # Разделение выборки на тренировочную и тестовую
    (train_data, test_data, train_labels, test_labels) = train_test_split(np.array(data),
                                                                          np.array(labels),
                                                                          test_size=test_size,
                                                                          random_state=seed)
    # Определение архитектуры сверточной НС
    # с двумя сверточными (Conv) и двумя полносвязными слоями (Dense)
    model = Sequential()
    # Сверточный слой с 32 фильтрами и ядром 3x3, ф-ия активации relu, размерность входных данных (32, 32, 3)
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
    # Слой субдискретизации для снижения размерности в 2 раза по каждой оси
    model.add(MaxPool2D(pool_size=(2, 2)))
    # Слой исключения нейронов для предотвращения переобучения
    model.add(Dropout(0.3))
    # Второй аналогичный сверточный слой
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    # Слой, переводящий многомерные данные в одномерный массив
    model.add(Flatten())
    # Два скрытых полносвязных слоя с Dropout и выходной слой
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))

    # Компилируем модель, используя Adam как оптимизатор и категориальную
    # кросс-энтропию в качестве функции потерь
    print("[INFO] Компиляция модели...")
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model.summary()

    # Обучаем модель, epochs - количество эпох, batch_size - контролирует размер пакета данных для передачи по сети
    print("[INFO] Обучение модели...")
    history = model.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=15, batch_size=32)
    print("[INFO] Оценка модели...")
    estimate_model(model, test_data, test_labels, class_names, history)
    print("[INFO] Сериализация модели...")
    model.save(model_path)


def predict(class_names, model_name='FFNN'):
    """Распознавание изображения с помощью модели

    Параметры:
    class_names (list): список названий классов
    model_name (str): тип модели (FFNN или CNN)

    """
    # Загрузка выбранной модели
    if model_name == 'FFNN':
        model = keras.models.load_model(MODEL_FFNN_PATH)
    elif model_name == 'CNN':
        model = keras.models.load_model(MODEL_CNN_PATH)
    else:
        print("Error: No such model.")
        return

    print("[INFO] Распознавание тестовых изображений...")
    # Цикл по изображениям в тестовой директории
    for picture in os.listdir(TEST_PATH):
        path = os.path.join(TEST_PATH, picture)
        # Подготавливаем изображение аналогично prepare_sample.py
        image = cv2.imread(path)
        image_mod = cv2.resize(image, FIXED_SIZE)
        if model_name == 'FFNN':
            image_mod = image_mod.flatten()
        image_mod = image_mod / 255.0

        # Распознаём изображение
        preds = model.predict(np.array([image_mod]))[0]
        prediction = preds.argmax(axis=0)
        # Печать текстового обозначения класса на картинке
        label = "{}: {:.2f}%".format(class_names[prediction], preds[prediction] * 100)
        cv2.putText(image, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Вывод изображения
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        if model_name == 'FFNN':
            cv2.imwrite(r'data\output\test_FFNN\{}'.format(picture), image)
        elif model_name == 'CNN':
            cv2.imwrite(r'data\output\test_CNN\{}'.format(picture), image)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    data_FFNN, data_CNN, labels = get_data()
    class_names = os.listdir(TRAIN_PATH)
    #create_model_FFNN(data_FFNN, labels, class_names, TEST_SIZE, SEED, MODEL_FFNN_PATH)
    predict(class_names, 'FFNN')
    #create_model_CNN(data_CNN, labels, class_names, TEST_SIZE, SEED, MODEL_CNN_PATH)
    predict(class_names, 'CNN')
