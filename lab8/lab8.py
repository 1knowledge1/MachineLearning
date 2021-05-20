import warnings
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Dropout, Lambda
from tensorflow.keras.models import Model
from prepare_sample import *

BATCH_SIZE = 16

def get_data():
    """Получить данные из файлов .h5

    Возвращает:
    data (list): список данных
    labels (list): список меток

    """
    h5f_data= h5py.File(H5_DATA, 'r')
    h5f_label = h5py.File(H5_LABELS, 'r')
    data = np.array(h5f_data['dataset_1'])
    labels = np.array(h5f_label['dataset_1'])
    h5f_data.close()
    h5f_label.close()
    return data, labels


def create_dense_ae():
    # Размерность кодированного представления
    encoding_dim = 250
    autoencoder = Sequential(name='autoencoder')
    autoencoder.add(Input(shape=(28, 28, 1)))
    autoencoder.add(Flatten())
    autoencoder.add(Dense(encoding_dim, activation='relu'))

    autoencoder.add(Dense(28 * 28, activation='sigmoid'))
    autoencoder.add(Reshape((28, 28, 1)))
    return autoencoder
    #
    # # Энкодер
    # # Входной плейсхолдер
    # # 28, 28, 1 - размерности строк, столбцов, фильтров одной картинки, без батч-размерности
    # input_img = Input(shape=(28, 28, 1))
    # # Вспомогательный слой решейпинга
    # flat_img = Flatten()(input_img)
    # # Кодированное полносвязным слоем представление
    # encoded = Dense(encoding_dim, activation='relu')(flat_img)
    #
    # # Декодер
    # # Раскодированное другим полносвязным слоем изображение
    # input_encoded = Input(shape=(encoding_dim,))
    # flat_decoded = Dense(28 * 28, activation='sigmoid')(input_encoded)
    # decoded = Reshape((28, 28, 1))(flat_decoded)
    #
    # # Модели, в конструктор первым аргументом передаются входные слои, а вторым выходные слои
    # # Другие модели можно так же использовать как и слои
    # encoder = Model(input_img, encoded, name="encoder")
    # decoder = Model(input_encoded, decoded, name="decoder")
    # autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    # return encoder, decoder, autoencoder


def create_denoising_model(autoencoder):
    def add_noise(x):
        noise_factor = 0.3
        x = x + K.random_normal(x.get_shape(), 0.5, noise_factor)
        x = K.clip(x, 0., 1.)
        return x

    noiser = Sequential()
    noiser.add(Input(batch_shape=(BATCH_SIZE, 28, 28, 1)))
    noiser.add(Lambda(add_noise))
    denoiser_model = Model(noiser.inputs, autoencoder(noiser.output), name="denoiser")

    # input_img = Input(batch_shape=(BATCH_SIZE, 28, 28, 1))
    # noised_img = Lambda(add_noise)(input_img)
    # noiser = Model(input_img, noised_img, name="noiser")
    # denoiser_model = Model(input_img, autoencoder(noiser(input_img)), name="denoiser")
    return noiser, denoiser_model


def estimate_model(history):
    """Оценка модели с выводом графика потерь и точности

    Параметры:
    model (): модель НС
    test_data (list): список тестовых данных
    test_labels (list): список меток соответствующих тестовым данным
    history ():

    """

    # Строим графики потерь и точности
    N = history.epoch
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, history.history["loss"], label="train_loss")
    plt.plot(N, history.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def predict():
    noiser = tf.keras.models.load_model(NOISER_MODEL_PATH)
    autoencoder = tf.keras.models.load_model(AUTOENCODER_MODEL_PATH)

    for picture in os.listdir(TEST_PATH):
        path = os.path.join(TEST_PATH, picture)
        # Подготавливаем изображение аналогично prepare_sample.py
        image = cv2.imread(path, 0)
        image = cv2.resize(image, FIXED_SIZE)

        image = np.array(image) / 255.0
        image = np.reshape(image, (28, 28, 1))


        noised_imgs = noiser.predict(np.array([image]))[0]
        decoded_imgs = autoencoder.predict(np.array([noised_imgs]))[0]

        # encoded_imgs = encoder.predict(np.array([noised_imgs]))[0]
        # decoded_imgs = decoder.predict(np.array([encoded_imgs]))[0]

        noised_imgs = (noised_imgs * 255).astype("uint8")
        decoded_imgs = (decoded_imgs * 255).astype("uint8")
        output = np.hstack((noised_imgs, decoded_imgs))
        cv2.imwrite(r'data\output\results\{}'.format(picture), output)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    tf.random.set_seed(0)
    data, labels = get_data()
    (train_data, test_data, train_labels, test_labels) = train_test_split(np.array(data),
                                                                          np.array(labels),
                                                                          test_size=0.2,
                                                                          random_state=17)
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')
    train_data = np.reshape(train_data, (len(train_data), 28, 28, 1))
    test_data = np.reshape(test_data, (len(test_data), 28, 28, 1))


    # encoder, decoder, autoencoder = create_dense_ae()
    autoencoder = create_dense_ae()
    noiser, denoiser_model = create_denoising_model(autoencoder)
    denoiser_model.compile(optimizer='adam', loss='binary_crossentropy')
    denoiser_model.summary()
    history = denoiser_model.fit(train_data, train_data,
                    epochs=40,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    validation_data=(test_data, test_data))

    estimate_model(history)
    noiser.save(NOISER_MODEL_PATH)
    autoencoder.save(AUTOENCODER_MODEL_PATH)
    predict()


