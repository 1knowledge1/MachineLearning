import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE


if __name__ == '__main__':
    # Извлекаем датасет с фичами из лаб. 3
    h5f_data = h5py.File('data.h5', 'r')
    h5f_label = h5py.File('labels.h5', 'r')
    global_features = np.array(h5f_data['dataset_1'])
    global_labels = np.array(h5f_label['dataset_1'])
    h5f_data.close()
    h5f_label.close()

    # Извлекаем статьи из бинарного файла из лаб. 4
    with open('wiki_articles.bin', 'rb') as fin:
        wiki_articles = pickle.load(fin)
    # Извлекаем фичи из статей с помощью TF-IDF
    vectorizer = TfidfVectorizer(decode_error='replace', min_df=5, max_df=0.8,
                                 sublinear_tf=True, use_idf=True, stop_words={'english'})
    tf_idf_weight = vectorizer.fit_transform(wiki_articles)

    # Применяем метод главных компонент (PCA) для снижения размерности массива фич до 2
    # Исходная размерность 532, после применения PCA -> 2
    X_pca = PCA(n_components=2, random_state=17)
    components = X_pca.fit_transform(global_features)
    # Выводим данные на график с помощью scatter
    plt.figure()
    # Аргументы: x - значения первой компоненты фич, y - второй, с - цвета
    plt.scatter(components[:, 0], components[:, 1], c=global_labels)
    plt.title('PCA. Pokemon (lab3)')
    plt.show()

    # Применяем метод t-SNE для нелинейного снижения размерности до 2
    # Аргументы:
    # perplexity связан с количеством ближайших соседей и подбирается в ручную, обычно от 5 до 50
    # n_iter - количество итераций, verbose - уровень детализации, random_state - параметр для
    # генератора случайных чисел, для воспроизводимости результатов
    tsne = TSNE(n_components=2, perplexity=15, n_iter=500, verbose=1, random_state=13)
    result = tsne.fit_transform(global_features)
    plt.figure()
    plt.scatter(result[:, 0], result[:, 1], c=global_labels)
    plt.title('TSNE. Pokemon (lab3)')
    plt.show()

    # Применяем метод усеченного SVD, так как он работает подобно PCA, но с возможностью работы
    # на разряженной матрице (tf_idf_weight)
    X_pca = TruncatedSVD(n_components=2, random_state=13)
    components = X_pca.fit_transform(tf_idf_weight)
    # Выводим на график.  n - количество статей в теме, colors - цвета меток, labels - темы статей
    n = 15
    colors = ['r', 'g', 'b']
    labels = ['Mathematicians', ' History of Asia', 'Media']
    plt.figure()
    for i in range(0, 3):
        plt.scatter(components[i*n:(i+1)*n, 0], components[i*n:(i+1)*n, 1], c=colors[i], label=labels[i])
    plt.legend()
    plt.title('PCA. Wikipedia (lab4)')
    plt.show()

    # Применяем метод t-SNE для снижения размерности до 2
    tsne = TSNE(n_components=2, perplexity=25, n_iter=700, verbose=1, random_state=13)
    result = tsne.fit_transform(tf_idf_weight)
    plt.figure()
    for i in range(0, 3):
        plt.scatter(result[i*n:(i+1)*n, 0], result[i*n:(i+1)*n, 1], c=colors[i], label=labels[i])
    plt.legend()
    plt.title('TSNE. Wikipedia (lab4)')
    plt.show()
