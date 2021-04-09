import warnings
import glob
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from feature_extractor import *
warnings.filterwarnings('ignore')


num_trees = 120                 # Количество деревьев для RFC
test_size = 0.10                # Размер тестовой выборки
seed = 7                        # Параметр для рандомизации
test_path = 'data/test'         # Путь до тестового датасета


if __name__ == "__main__":
    train_labels = os.listdir(train_path)
    train_labels.sort()

    # Извлекаем датасет с извелеченными фичами
    h5f_data = h5py.File(h5_data, 'r')
    h5f_label = h5py.File(h5_labels, 'r')
    global_features = np.array(h5f_data['dataset_1'])
    global_labels = np.array(h5f_label['dataset_1'])
    h5f_data.close()
    h5f_label.close()

    models = []
    models.append(('LR', LogisticRegression(random_state=seed)))        # Логистическая регрессия
    models.append(('CART', DecisionTreeClassifier(random_state=seed)))  # Классификатор дерево решений
    models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed)))    # Случайный лес
    models.append(('NB', GaussianNB()))     # Наивный байесовский классификатор
    models.append(('SVM', SVC(random_state=seed)))      # Метод опорных векторов

    # Разеляем выборку на тестовую и тренировочную
    (trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                              np.array(global_labels),
                                                                                              test_size=test_size,
                                                                                              random_state=seed)
    # Результаты кроссвалидации
    accuracy = []
    recall = []
    precision = []
    names = [] # Названия классификаторов для вывода на график
    metrics = {'Аккуратность': accuracy, 'Полнота': recall, 'Точность': precision}

    for name, model in models:
        kfold = KFold(n_splits=3, shuffle=True, random_state=seed)
        cv_results_acc = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring="accuracy")
        cv_results_prec = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring="precision_macro")
        kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
        cv_results_rec = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring="recall_macro")
        accuracy.append(cv_results_acc)
        precision.append(cv_results_prec)
        recall.append(cv_results_rec)
        names.append(name)
        # model.fit(trainDataGlobal, trainLabelsGlobal)
        # print(classification_report(testLabelsGlobal, model.predict(testDataGlobal), target_names=train_labels))

    modelRF = models[2][1]
    modelRF.fit(trainDataGlobal, trainLabelsGlobal)
    print(classification_report(testLabelsGlobal, modelRF.predict(testDataGlobal), target_names=train_labels))

    # Выводим "ящик с усами" для метрик
    for title, values in metrics.items():
        fig = pyplot.figure()
        fig.suptitle(title)
        ax = fig.add_subplot(111)
        pyplot.boxplot(values)
        ax.set_xticklabels(names)
    pyplot.show()

    # Создаем модель, классификатор случайный лес
    clf = RandomForestClassifier(n_estimators=num_trees, random_state=seed)

    # Тренировка модели на тренировчной выборке
    clf.fit(trainDataGlobal, trainLabelsGlobal)

    # Цикл по всем тестовым изображениям
    for file in glob.glob(test_path + "/*.jpg"):
        # Читаем и изменяем размер изображения
        image = cv2.imread(file)
        image = cv2.resize(image, fixed_size)
        # Вычисляем фичи
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick = fd_haralick(image)
        fv_histogram = fd_histogram(image)
        # Объединяем фичи в одну глобальную
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
        # Классифицировать картинку по фичам
        prediction = clf.predict(global_feature.reshape(1, -1))[0]
        # Разместить текст с названием класса на изображении
        cv2.putText(image, train_labels[prediction], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        # display the output image
        pyplot.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pyplot.show()
