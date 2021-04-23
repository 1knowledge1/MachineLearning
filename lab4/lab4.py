import pickle
import os
import pandas as pd
import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram


def download_articles(articles_names):
    """Загрузить статьи из Википедии в список и сохранить его локально

    Параметры:
    articles_names (list): список названий статей

    Возвращает:
    list: список статей

    """
    wiki_articles = []
    for name in articles_names:
        wiki_articles.append(wikipedia.page(name).content)
        print(name + ' has been downloaded.')
    with open('wiki_articles.bin', 'wb') as fout:
        pickle.dump(wiki_articles, fout)
    return wiki_articles


def read_articles():
    """Загрузить статьи из файла в список

        Возвращает:
        list: список статей

    """
    with open('wiki_articles.bin', 'rb') as fin:
        wiki_articles = pickle.load(fin)
    return wiki_articles


if __name__ == '__main__':
    # Список статей Википедии по темам: Mathematicians, History of Asia, Media
    articles = ['Joseph-Louis Lagrange', 'Archimedes', 'Euclid', 'Muhammad ibn Musa al-Khwarizmi', 'Leonhard Euler',
                'Pierre-Simon Laplace', 'Carl Friedrich Gauss', 'Emmy Noether', 'Kurt Gödel',
                'Gottfried Wilhelm Leibniz', 'Isaac Newton', 'Pierre de Fermat',
                'Andrey Kolmogorov', 'René Descartes', 'John Forbes Nash Jr.',

                'History of Asia', 'Achaemenid Empire', 'Sasanian Empire',
                'History of Indian influence on Southeast Asia', 'Zhou dynasty', 'Qin dynasty',
                'Han dynasty', 'Jin dynasty (266–420)', 'Mongol Empire', 'Medieval India',
                'History of East Asia', 'History of China', 'Goryeo', 'History of Japan', 'Qing dynasty',

                'Mass media', 'Broadcasting', 'Journalism', 'Social media', 'Publishing', 'Radio', 'Television',
                'Newspaper', 'Public speaking', 'Media (communication)', 'Digital media', 'Media independence',
                'Mobile media', 'News media', 'State media']

    # Скачиваем и сериализуем список статей, чтобы не скачивать его, каждый раз,
    # или десериализуем список статей, если они уже были скачены и записаны в бинарный файл
    if os.path.exists('./wiki_articles.bin'):
        wiki_articles = read_articles()
    else:
        wiki_articles = download_articles(articles)

    # Извлекаем фичи с помощью TF-IDF: для каждого слова в статье вычисляется параметр tfidf, зависящий от того,
    # как часто встречается это слово
    # Задаем параметры: если слово встречается меньше 5 раз, или оно встречается чаще 80% всех слов статьи,
    # то его не учитываем
    vectorizer = TfidfVectorizer(decode_error='replace', min_df=5, max_df=0.8,
                                 sublinear_tf=True, use_idf=True, stop_words={'english'})
    tf_idf_weight = vectorizer.fit_transform(wiki_articles)

    # Определяем число кластеров K с помощь elbow метода:
    # точка изгиба на графике покажет оптимальное k.
    # Предполагаем, что k лежит в интервале от 2 до 6,
    # и кластеризуем по методу k-средних для каждого k, макс. число итераций = 200
    K = range(2, 6)
    sum_of_squared_distances = []
    for k in K:
        km = KMeans(n_clusters=k, max_iter=200)
        km = km.fit(tf_idf_weight)
        sum_of_squared_distances.append(km.inertia_)
    # Выводим график
    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

    # В точке изгиба значение k = 3. Как и ожидалось, число кластеров совпадает с количеством тем статей
    # Выполняем кластеризацию по алгоритму k-средних и выводим результат
    true_k = 3
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=200)
    model.fit(tf_idf_weight)
    labels = model.labels_
    wiki_cl = pd.DataFrame(list(zip(articles, labels)), columns=['title', 'cluster'])
    print('KMeans')
    print(wiki_cl.sort_values(by=['cluster']))

    # Выполняем кластеризацию по алгоритму MiniBatchKMeans и выводим результат
    model = MiniBatchKMeans(n_clusters=3, init='random', max_iter=100, n_init=5)
    model.fit(tf_idf_weight)
    labels = model.labels_
    wiki_cl = pd.DataFrame(list(zip(articles, labels)), columns=['title', 'cluster'])
    print('\nMiniBatchKMeans')
    print(wiki_cl.sort_values(by=['cluster']))

    # Выполняем кластеризацию по алгоритму DBSCAN и выводим результат
    model = DBSCAN(eps=1.131, min_samples=8)
    model.fit(tf_idf_weight)
    labels = model.labels_
    wiki_cl = pd.DataFrame(list(zip(articles, labels)), columns=['title', 'cluster'])
    print('\nDBSCAN')
    print(wiki_cl.sort_values(by=['cluster']))

    # Для построения матрицы сходства используем метод Уорда, так как  он
    # лучше всего подходит для кластеризации текстов
    dist = 1 - cosine_similarity(tf_idf_weight)
    linkage_matrix = ward(dist)
    # По определенной матрицы строим дендрограмму с заданными параметрами и выводим
    fig, ax = plt.subplots(figsize=(8, 8))
    ax = dendrogram(linkage_matrix, orientation="right", labels=articles)
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.tight_layout()
    plt.show()
    plt.close()
