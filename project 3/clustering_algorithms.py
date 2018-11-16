from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import numpy as np

from graphing import plot_confusion_matrix


def run_clustering_algo_single(X, y, algorithm, classifier, as_int=False, neighbors=None):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                        test_size=0.2)  # produces good shuffled train and test sets
    if as_int:
        X_train = X_train.astype('int')
        X_test = X_test.astype('int')
        y_train = y_train.astype('int')
        y_test = y_test.astype('int')

    algorithm.fit(X_train)  # does the fitting (training)

    # add visualization here

    if type(algorithm) == KMeans:
        # add conversions
        transformed_X_train = algorithm.fit_transform(X_train)
        transformed_X_test = algorithm.fit_transform(X_test)
        classifier.fit(transformed_X_train, y_train)
        y_pred = classifier.predict(transformed_X_test)
        cm = confusion_matrix(y_test, y_pred)
        if neighbors:
            plot_confusion_matrix(cm, ["Class 1", "Class 2"], "Kmeans", title="Breast Cancer K Means (" + str(neighbors) + " neighbors) Neural Net Confusion Matrix")

    if type(algorithm) == GaussianMixture:
        transformed_X_train = algorithm.fit_predict(X_train)
        transformed_X_train = np.array([[x] for x in transformed_X_train])
        classifier.fit(transformed_X_train, y_train)

        transformed_X_test = algorithm.predict(X_test)
        transformed_X_test = np.array([[x] for x in transformed_X_test])

        y_pred = classifier.predict(transformed_X_test)
        cm = confusion_matrix(y_test, y_pred)
        if neighbors:
            plot_confusion_matrix(cm, ["Class 1", "Class 2"], "GMM", title="Breast Cancer GMM with EM (" + str(neighbors) + " neighbors) Confusion Matrix")

    if type(algorithm) == PCA:
        # import ipdb; ipdb.set_trace()
        transformed_X_train = algorithm.fit_transform(X_train)
        classifier.fit(transformed_X_train, y_train)

        transformed_X_test = algorithm.transform(X_test)
        y_pred = classifier.predict(transformed_X_test)

        cm = confusion_matrix(y_test, y_pred)
        if neighbors:
            plot_confusion_matrix(cm, ["Class 1", "Class 2"], "PCA", title="Breast Cancer PCA (" + str(neighbors) + " neighbors) Neural Net Confusion Matrix")

    confidence = algorithm.score(X_test, y_test)  # (test)
    test_accuracy = classifier.score(transformed_X_test, y_test)

    return confidence, test_accuracy
