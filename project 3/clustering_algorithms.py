from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from graphing import plot_confusion_matrix, plot_cross_section, plot_learning_curve


def run_clustering_algo_single(X, y, algorithm, classifier, as_int=False, neighbors=None, cross_sections=False):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                        test_size=0.2)  # produces good shuffled train and test sets
    if as_int:
        X_train = X_train.astype('int')
        X_test = X_test.astype('int')
        y_train = y_train.astype('int')
        y_test = y_test.astype('int')

    algorithm.fit(X_train)  # does the fitting (training)
    best_test_accuracy = -1

    for i in range(10):
        if type(algorithm) == KMeans:
            # add conversions
            transformed_X_train = algorithm.fit_transform(X_train)

            transformed_X_test = algorithm.transform(X_test)
            title = "Kmeans"

        if type(algorithm) == GaussianMixture:
            transformed_X_train = algorithm.fit_predict(X_train)
            transformed_X_train = np.array([[x] for x in transformed_X_train])

            transformed_X_test = algorithm.predict(X_test)
            transformed_X_test = np.array([[x] for x in transformed_X_test])
            title = "GMM"
            import ipdb;
            ipdb.set_trace()

        if type(algorithm) == PCA:
            transformed_X_train = algorithm.fit_transform(X_train)

            transformed_X_test = algorithm.transform(X_test)
            title = "PCA"
            import ipdb;
            ipdb.set_trace()

        classifier.fit(transformed_X_train, y_train)
        confidence = algorithm.score(X_test, y_test)
        test_accuracy = classifier.score(transformed_X_test, y_test)
        y_pred = classifier.predict(transformed_X_test)
        cm = confusion_matrix(y_test, y_pred)
        if neighbors >= 2 and test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            plot_confusion_matrix(cm, ["Class 1", "Class 2"], title, title="Breast Cancer " + title + " (" + str(
                neighbors) + " neighbors) Neural Net Confusion Matrix")
            df = pd.DataFrame(transformed_X_train)
            c = df.corr().abs()
            s = c.unstack()
            so = s.sort_values(kind="quicksort")
            so = so[so != 1]
            max_cross_section = so.idxmax()
            min_cross_section = so.idxmin()
            for cross_section in [max_cross_section, min_cross_section]:
                plot_cross_section(transformed_X_train, cross_section, title, neighbors)
            # import ipdb; ipdb.set_trace()

    return confidence, test_accuracy


def run_PCA(X_train, X_test, y_train, y_test):
    algorithm = PCA(random_state=0, n_components=0.99)

    transformed_X_train = algorithm.fit_transform(X_train)
    df = pd.DataFrame()
    df['label'] = pd.Series([i[0] for i in y_train.tolist()])
    df['comp-one'] = transformed_X_train[:, 0]
    df['comp-two'] = transformed_X_train[:, 1]
    transformed_X_test = algorithm.transform(X_test)

    confidence = algorithm.score(X_test, y_test)
    import ipdb; ipdb.set_trace()
    return confidence, transformed_X_train, transformed_X_test, df


def run_Kmeans(X_train, X_test, y_train, y_test, experiment_number, dataset_name, neighbors=None):
    if neighbors:
        algorithm = KMeans(random_state=0, n_clusters=neighbors)
    else:
        algorithm = KMeans(random_state=0)

    # add conversions
    transformed_X_train = algorithm.fit_transform(X_train)
    transformed_X_test = algorithm.fit_transform(X_test)
    df = pd.DataFrame()
    title = "Kmeans"

    confidence = algorithm.score(X_test, y_test)
    inertia = algorithm.inertia_
    if neighbors >= 2:
        df['label'] = pd.Series([i[0] for i in y_train.tolist()])
        df['comp-one'] = transformed_X_train[:, 0]
        df['comp-two'] = transformed_X_train[:, 1]
        transformed_df = pd.DataFrame(transformed_X_train)
        c = transformed_df.corr().abs()
        s = c.unstack()
        so = s.sort_values(kind="quicksort")
        so = so[so != 1]
        max_cross_section = so.idxmax()
        min_cross_section = so.idxmin()
        for cross_section in [max_cross_section, min_cross_section]:
            plot_cross_section(transformed_X_train, cross_section, title, neighbors, experiment_number, dataset_name)

    return confidence, inertia, transformed_X_train, transformed_X_test, df


def run_GMM(X_train, X_test, y_train, y_test, neighbors=None):
    if neighbors:
        algorithm = GaussianMixture(random_state=0, n_components=neighbors)
    else:
        algorithm = GaussianMixture(random_state=0)

    # add conversions
    transformed_X_train = algorithm.fit_predict(X_train)
    transformed_X_train = np.array([[x] for x in transformed_X_train])

    transformed_X_test = algorithm.fit_predict(X_test)
    transformed_X_test = np.array([[x] for x in transformed_X_test])

    confidence = algorithm.score(X_test, y_test)

    return confidence, transformed_X_train, transformed_X_test


def run_Neural_Net(breast_cancer_data, path, algorithm, neighbors=None):
    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=4)
    breast_cancer_X_train, breast_cancer_X_test, breast_cancer_y_train, breast_cancer_y_test = breast_cancer_data

    classifier.fit(breast_cancer_X_train, breast_cancer_y_train)

    test_accuracy = classifier.score(breast_cancer_X_test, breast_cancer_y_test)

    y_pred = classifier.predict(breast_cancer_X_test)

    cm = confusion_matrix(breast_cancer_y_test, y_pred)
    if neighbors:
        title = "Breast Cancer " + algorithm + " (" + str(neighbors) + " neighbors) Neural Net Confusion Matrix"
    else:
        title = "Breast Cancer " + algorithm + " Neural Net Confusion Matrix"

    plot_confusion_matrix(cm, ["Class 1", "Class 2"], path, title=title)

    X = np.concatenate((breast_cancer_X_train, breast_cancer_X_test))
    y = np.concatenate((breast_cancer_y_train, breast_cancer_y_test))
    plot_learning_curve(classifier, path, "Breast Cancer " + algorithm + " Neural Net Learning Curve", X, y)
    return test_accuracy
