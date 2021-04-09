import glob
import warnings
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import classification_report, make_scorer, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from feature_extractor import *
warnings.filterwarnings('ignore')


num_trees = 100
test_size = 0.10
seed = 9
train_path = "data/train"
test_path = "data/test"
h5_data = 'data/data.h5'
h5_labels = 'data/labels.h5'
scoring = "accuracy"

originalclass = []
predictedclass = []

# Переопределение для score, чтобы напечатать отчет
def classification_report_with_accuracy_score(y_true, y_pred):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    return accuracy_score(y_true, y_pred)


if __name__ == "__main__":
    train_labels = os.listdir(train_path)
    train_labels.sort()

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    models = []
    models.append(('LR', LogisticRegression(random_state=seed)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier(random_state=seed)))
    models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed)))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(random_state=seed)))

    # variables to hold the results and names
    results_acc = []
    results_rec = []
    results_prec = []
    names = []

    # import the feature vector and trained labels
    h5f_data = h5py.File(h5_data, 'r')
    h5f_label = h5py.File(h5_labels, 'r')

    global_features_string = h5f_data['dataset_1']
    global_labels_string = h5f_label['dataset_1']

    global_features = np.array(global_features_string)
    global_labels = np.array(global_labels_string)

    h5f_data.close()
    h5f_label.close()

    # split the training and testing data
    (trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                              np.array(global_labels),
                                                                                              test_size=test_size,
                                                                                              random_state=seed)

    print("[STATUS] splitted train and test data...")
    print("Train data  : {}".format(trainDataGlobal.shape))
    print("Test data   : {}".format(testDataGlobal.shape))
    print("Train labels: {}".format(trainLabelsGlobal.shape))
    print("Test labels : {}".format(testLabelsGlobal.shape))

    for name, model in models:
        kfold = KFold(n_splits=3, shuffle=True, random_state=seed)
        cv_results_acc = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring="accuracy")
        cv_results_prec = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold,
                                          scoring="precision_macro")
        kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
        cv_results_rec = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring="recall_macro")
        results_acc.append(cv_results_acc)
        results_prec.append(cv_results_prec)
        results_rec.append(cv_results_rec)
        names.append(name)

    # Еще раз выполняем кроссвалидацию с лучшим из лучших (с кастомным scoring), чтобы напечатать отчет
    cross_val_score(RandomForestClassifier(random_state=seed), trainDataGlobal, trainLabelsGlobal,
                    cv=KFold(n_splits=5, shuffle=True, random_state=seed),
                    scoring=make_scorer(classification_report_with_accuracy_score))
    print(classification_report(originalclass, predictedclass, target_names=train_labels))

    # Выводим ящик с усами на одной картинке
    fig = pyplot.figure()
    ax1 = fig.add_subplot(131)
    ax1.set(title='accurancy')
    pyplot.boxplot(results_acc)
    ax1.set_xticklabels(names)

    ax2 = fig.add_subplot(133)
    pyplot.boxplot(results_rec)
    ax2.set(title='recall')
    ax2.set_xticklabels(names)

    ax3 = fig.add_subplot(132)
    pyplot.boxplot(results_prec)
    ax3.set(title='precision')
    ax3.set_xticklabels(names)
    pyplot.show()

    # create the model - Random Forests
    clf = RandomForestClassifier(n_estimators=num_trees, random_state=seed)

    # fit the training data to the model
    clf.fit(trainDataGlobal, trainLabelsGlobal)

    # loop through the test images
    for file in glob.glob(test_path + "/*.jpg"):
        # read the image
        image = cv2.imread(file)

        # resize the image
        image = cv2.resize(image, fixed_size)

        ####################################
        # Global Feature extraction
        ####################################
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick = fd_haralick(image)
        fv_histogram = fd_histogram(image)

        ###################################
        # Concatenate global features
        ###################################
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        # predict label of test image
        prediction = clf.predict(global_feature.reshape(1, -1))[0]

        # show predicted label on image
        cv2.putText(image, train_labels[prediction], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        # display the output image
        pyplot.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pyplot.show()
