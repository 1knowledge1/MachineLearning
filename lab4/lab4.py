import pickle
import pandas as pd
import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN


def download_articles(articles_names):
    """Загрузить статьи в список и сохранить его локально."""
    wiki_lst = []
    for article in articles_names:
        wiki_lst.append(wikipedia.page(article).content)
        print(article + ' has been downloaded.')
    with open('wiki_lst.bin', 'wb') as fout:
        pickle.dump(wiki_lst, fout)
    return wiki_lst


def read_articles():
    """Загрузить статьи из файла в список."""
    with open('wiki_lst.bin', 'rb') as fin:
        wiki_lst = pickle.load(fin)
    return wiki_lst


if __name__ == '__main__':
    articles = ['Joseph-Louis Lagrange', 'Archimedes', 'Euclid', 'Muhammad ibn Musa al-Khwarizmi', 'Leonhard Euler',
                'Pierre-Simon Laplace', 'Carl Friedrich Gauss', 'Emmy Noether', 'Kurt Gödel',
                'Gottfried Wilhelm Leibniz', 'Isaac Newton', 'Pierre de Fermat',
                'Andrey Kolmogorov', 'René Descartes', 'John Forbes Nash Jr.',

                'History of Asia', 'Achaemenid Empire', 'Sasanian Empire',
                'History of Indian influence on Southeast Asia', 'Zhou dynasty', 'Qin dynasty',
                'Han dynasty', 'Jin dynasty (266–420)', 'Mongol Empire', 'Medieval India',
                'History of East Asia', 'History of China', 'Goryeo', 'History of Japan', 'Qing dynasty',

                'Mass media', 'Broadcasting', 'Journalism', 'Social media', 'Publishing', 'Radio', 'Television',
                'Newspaper', 'Public speaking', 'Internet', 'Digital media', 'Blog', 'Mobile media',
                'News media', 'State media']

    # wiki_lst = download_articles(articles)
    wiki_lst = read_articles()

    vectorizer = TfidfVectorizer(decode_error='replace', min_df=5, max_df=0.8,
                                 sublinear_tf=True, use_idf=True, stop_words={'english'})
    tf_idf_weight = vectorizer.fit_transform(wiki_lst)

    print(vectorizer.vocabulary_)
    print(tf_idf_weight.shape)

    Sum_of_squared_distances = []
    K = range(2, 6)
    for k in K:
        km = KMeans(n_clusters=k, max_iter=200, n_init=10)
        km = km.fit(tf_idf_weight)
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()
     #_______________________________________________________________________________________________________________
    print('KMeans')
    true_k = 3
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=20)
    model.fit(tf_idf_weight)
    labels = model.labels_
    wiki_cl = pd.DataFrame(list(zip(articles, labels)), columns=['title', 'cluster'])
    print(wiki_cl.sort_values(by=['cluster']))

    from sklearn.decomposition import PCA
    pca = PCA(n_components=4, random_state=0)
    reduced_features = pca.fit_transform(tf_idf_weight.toarray())

    # reduce the cluster centers to 2D
    reduced_cluster_centers = pca.transform(model.cluster_centers_)
    plt.scatter(reduced_features[:,0], reduced_features[:,1], c=model.predict(tf_idf_weight))
    plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')

    print('')
    print('MiniBatchKMeans')

    MiniBatch_model = MiniBatchKMeans(n_clusters=3, init='random', max_iter=100, batch_size=100, n_init=5)
    MiniBatch_model.fit(tf_idf_weight)
    MiniBatch_labels = MiniBatch_model.labels_
    MiniBatch_wiki_cl = pd.DataFrame(list(zip(articles, MiniBatch_labels)), columns=['title', 'cluster'])
    print(MiniBatch_wiki_cl.sort_values(by=['cluster']))

    print('')
    print('DBSCAN')

    DBSCAN_model = DBSCAN(eps=1.189, min_samples=10)
    DBSCAN_model.fit(tf_idf_weight)
    DBSCAN_labels = DBSCAN_model.labels_
    DBSCAN_wiki_cl = pd.DataFrame(list(zip(articles, DBSCAN_labels)), columns=['title', 'cluster'])
    print(DBSCAN_wiki_cl.sort_values(by=['cluster']))

    # from sklearn.neighbors import NearestNeighbors
    # from matplotlib import pyplot as plt
    # import numpy as np
    #
    # neighbors = NearestNeighbors(n_neighbors=10)
    # neighbors_fit = neighbors.fit(tf_idf_weight)
    # distances, indices = neighbors_fit.kneighbors(tf_idf_weight)
    #
    # distances = np.sort(distances, axis=0)
    # distances = distances[:,1]
    # plt.plot(distances)
    # plt.show()

    from sklearn.metrics.pairwise import cosine_similarity

    dist = 1 - cosine_similarity(tf_idf_weight)
    from scipy.cluster.hierarchy import ward, dendrogram

    linkage_matrix = ward(dist)  # define the linkage_matrix using ward clustering pre-computed distances

    fig, ax = plt.subplots(figsize=(15, 20))  # set size
    ax = dendrogram(linkage_matrix, orientation="right", labels=articles)

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')

    plt.tight_layout()  # show plot with tight layout

    # uncomment below to save figure
    # plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters
    plt.show()
    plt.close()