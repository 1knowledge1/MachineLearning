import pandas as pd
import re
import random
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import gensim.downloader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM


def prepare(text, stemmer, stop_words):
    """Подготовка текста: обработка стеммером, удаление стопслов

    Параметры:
    text (str): тект, который будет обработан
    stemmer (stemmer): стеммер
    stop_words (list): список стопслов

    Возвращает:
    (str): обработанный текст

    """
    # Заменяем большие буквы, удаляем спецсимволы
    text = text.lower()
    text = re.sub('[^a-zA-z0-9\s]', '', text)
    # Разделяем текст на слова
    token_words = word_tokenize(text)
    stem_sentence = []
    # Добавляем слово, обработанное стеммером, если его нет
    # в списке стопслов
    for token in token_words:
        if token not in stop_words:
            stem_sentence.append(stemmer.stem(token))
    return " ".join(stem_sentence)


def predict(text, emotion_code):
    """Распознавание эмоций

    Параметры:
    text (str): текст
    emotion_code (int): тональность: 0 - negative, 1 - positive

    """
    testX = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=45)
    score = model.predict([testX])[0]
    if emotion_code:
        emotion = 'Positive'
    else:
        emotion = 'Negative'
    if score < 0.5:
        prediction = 'Negative'
    else:
        prediction = 'Positive'
    print("Emotion: {1}, Prediction: {2}, Text: {0}, Score: {3}". format(text, emotion, prediction, float(score)))


if __name__ == '__main__':
    # Для воспроизводимости результатов зададим seed
    random.seed(42)

    # Объединяем файлы в один датасет
    files = ['data/amazon_cells_labelled.csv', 'data/imdb_labelled.csv', 'data/yelp_labelled.csv']
    dataset = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(dataset)
    # Инициализируем список частых слов (стопслов) и стеммер для получения основ слов
    stop_words = stopwords.words("english")
    stemmer = SnowballStemmer("english")
    # Обрабатываем каждую запись в датасете с помощью ф-ии prepare
    dataset.text = dataset.text.apply(lambda sentence: prepare(sentence, stemmer, stop_words))
    print(dataset)
    # Разделение выборки на тренировочную и тестовую
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=45)

    # Загружаем предобученную модель с векторным представлением слов, эта матрица будет
    # использоваться в качестве весов в Embedding слое
    glove_vectors = gensim.downloader.load("glove-twitter-25")
    # Для представляения текста в виде последовательности целых чисел применяем Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data.text)
    # Количество слов в датасете
    vocab_size = len(tokenizer.word_index) + 1

    # Ф-ия texts_to_sequences() представляет текст в виде вектора целых чисел. Для корректной работы
    # НС необходимо, чтобы векторы были одной размерности, поэтому используется ф-ия pad_sequences()
    trainX = pad_sequences(tokenizer.texts_to_sequences(train_data.text), maxlen=45)
    testX = pad_sequences(tokenizer.texts_to_sequences(test_data.text), maxlen=45)
    # Получаем метки классов
    trainY = np.array(train_data.emotion.tolist())
    trainY = trainY.reshape(-1, 1)
    testY = np.array(test_data.emotion.tolist())
    testY = testY.reshape(-1, 1)
    # Размерность вектора, которым кодируется слово
    VECTOR_SIZE = glove_vectors.vector_size
    # Получаем матрицу весов для нашего датасета, путем выбора из матрицы glove_vectors только
    # тех слов, которые есть в нашем датасете
    embedding_matrix = np.zeros((vocab_size, VECTOR_SIZE))
    for word, i in tokenizer.word_index.items():
        if word in glove_vectors:
            embedding_matrix[i] = glove_vectors[word]

    # Определяем архитектуру модели рекурентной НС
    model = Sequential()
    # Добавляем Embedding слой для работы с текстом
    model.add(Embedding(vocab_size, VECTOR_SIZE, weights=[embedding_matrix], input_length=45, trainable=False))
    model.add(Dropout(0.5))
    # Добавляем слой LSTM
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    # Добавляем выходной слой
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    # Компилируем модель
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    # Обучаем модель
    history = model.fit(trainX, trainY, batch_size=32, epochs=27, validation_split=0.1, verbose=1)
    # Распознавание эмоций с помощью модели на тестовой выборке
    predicted = model.predict(testX, verbose=1, batch_size=32)
    predicted = [0 if prediction < 0.5 else 1 for prediction in predicted]
    # Оценка полноты, точности и аккуратности модели
    print(classification_report(test_data.emotion.tolist(), predicted))

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

    # Выборка из 10 предложений для проверки работы НС
    sentences = [["So Far So Good!", 1],
                 ['Works great!', 1],
                 ['I bought it for my mother and she had a problem with the battery.', 0],
                 ['Talk about USELESS customer service.', 0],
                 ["No ear loop needed, it's tiny and the sound is great.", 1],
                 ['As they say in Canada, This is the fun game, aye', 1],
                 ['The story itself is just predictable and lazy.', 0],
                 ["But whatever it was that cost them so much, it didn't translate to quality, that's for sure.", 0],
                 ["Long, whiny and pointless.", 0],
                 ["He is an amazing film artist, one of the most important whoever lived.", 1]]
    # Работа НС на выборке с выводом результата
    for sentence in sentences:
        predict(sentence[0], sentence[1])

    # Для визуализации слов используем scatter график. Сначала снизим размерность используя t-SNE
    tsne = TSNE(n_components=2, random_state=7)
    vectors = tsne.fit_transform(np.asarray(embedding_matrix))
    x_coordinate = [vector[0] for vector in vectors]
    y_coordinate = [vector[1] for vector in vectors]

    # Инициализируем scatter график
    plt.figure(figsize=(10, 10))
    plt.scatter(x_coordinate, y_coordinate, c='r', s=10)
    plt.title('2D t-SNE projection')
    # Выбираем 10 слов и выводим на график
    labels = [word for word in tokenizer.word_index.items()]
    indices = list(range(len(labels)))
    indices = random.sample(indices, 10)
    word_number = 0
    for i in indices:
        plt.annotate(labels[i][0], (x_coordinate[i], y_coordinate[i]), fontsize=10+word_number, c='b')
        word_number += 1
    plt.show()
