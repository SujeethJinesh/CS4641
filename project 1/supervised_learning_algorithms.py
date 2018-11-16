import numpy as np
import pandas as pd
from graphing import plot, plot_learning_curve, plot_confusion_matrix, plot_contours

from sklearn import model_selection, preprocessing, tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
import subprocess


def run_supervised_algo_single(data, label_cols, classifier, as_int=False):
    X = np.array(data.drop(label_cols, 1))  # X is all of our features
    X = preprocessing.scale(X.astype(float)).astype(float)  # scale our features

    y = np.array(data[label_cols])  # our y values

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                        test_size=0.2)  # produces good shuffled train and test sets
    if as_int:
        X_train = X_train.astype('int')
        X_test = X_test.astype('int')
        y_train = y_train.astype('int')
        y_test = y_test.astype('int')

    classifier.fit(X_train, y_train)  # does the prediction (training)

    # import ipdb; ipdb.set_trace()

    if type(classifier) == DecisionTreeClassifier:
        tree.export_graphviz(classifier)
        subprocess.call('dot -Tpng tree.dot -o tree.png', shell=True)
        plot_learning_curve(classifier, "Lung Cancer Decision Tree Learning Curve", X, y)

        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, ["Class 1", "Class 2", "Class 3"], title="Lung Cancer Decision Tree Confusion Matrix")

    if type(classifier) == MLPClassifier:
        plot_learning_curve(classifier, "Lung Cancer Neural Net Learning Curve", X, y)

        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, ["Class 1", "Class 2", "Class 3"], title="Lung Cancer Neural Net Confusion Matrix")

    if type(classifier) == AdaBoostClassifier:
        plot_learning_curve(classifier, "Lung Cancer AdaBoost Learning Curve", X, y)

        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, ["Class 1", "Class 2", "Class 3"], title="Lung Cancer AdaBoost Confusion Matrix")

    if type(classifier) == SVC:
        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test.astype(float), y_pred)
        plot_confusion_matrix(cm, ["Class 1", "Class 2", "Class 3"], title="Lung Cancer SVM Confusion Matrix")

    if type(classifier) == KNeighborsClassifier:
        plot_learning_curve(classifier, "Lung Cancer KNN Learning Curve", X, y)

        y_pred = classifier.predict(X_test)
        cm = confusion_matrix(y_test.astype(float), y_pred)
        plot_confusion_matrix(cm, ["Class 1", "Class 2", "Class 3"], title="Lung Cancer KNN Confusion Matrix")

    confidence = classifier.score(X_test, y_test)  # (test)

    return confidence, X, y


def run_supervised_algo_multi(data, label_col, classifier, as_int=False, graph=False):
    accuracies = []
    for key in data.columns:
        if key != label_col:
            new_data = pd.concat([data[key], data[label_col]], axis=1)
            # import ipdb; ipdb.set_trace()
            accuracy, X, y = run_supervised_algo_single(new_data, label_col, classifier, as_int=as_int)
            accuracies.append((key, accuracy, X, y))

    accuracies = sorted(accuracies, key=lambda x: (-x[1], x[0]))
    j = True
    for key, accuracy, X, y in accuracies:
        print(key, " accuracy was: ", accuracy)
        if graph and j:
            j = False
            plot(X, y)
