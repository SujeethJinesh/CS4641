import numpy as np
import pandas as pd
from graph import plot

from sklearn import model_selection, preprocessing


def run_supervised_algo_single(data, label_col, classifier, as_int=False):
    X = np.array(data.drop([label_col], 1))  # X is all of our features
    X = preprocessing.scale(X.astype(float)).astype(float)  # scale our features

    y = np.array(data[label_col])  # our y values

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                        test_size=0.2)  # produces good shuffled train and test sets

    if as_int:
        X_train = X_train.astype('int')
        X_test = X_test.astype('int')
        y_train = y_train.astype('int')
        y_test = y_test.astype('int')

    classifier.fit(X_train, y_train)  # does the prediction (training)

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
