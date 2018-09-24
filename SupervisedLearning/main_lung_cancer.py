from operator import itemgetter

from sklearn import svm, tree

from data_processing import getCleanData
from supervised_learning_algorithms import run_supervised_algo_single, run_supervised_algo_multi
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

import statistics as s
import pandas as pd
from pandas.tools.plotting import table
import matplotlib.pyplot as plt
import subprocess
import numpy as np

def main():
    # attributes
    attrs = range(1, 58)

    binary_one_hot_map = {
    }

    # might want to consider separating people by age band, might make it easier to identify relationships.
    normalize_list = []  # range(1, 58)

    multi_one_hot_list = [
    ]

    cancer_path = 'data/lung_cancer/lung-cancer.data'

    cancer_data = getCleanData(cancer_path, attrs, binary_one_hot_map=binary_one_hot_map, normalize_list=normalize_list,
                               multi_one_hot_list=multi_one_hot_list, missing_data_marker='?', to_impute=False, normalizer_type='min_max')

    # CheckList
    # 1. Decision Tree with pruning
    # 2. Neural Network
    # 3. Boosted Decision Tree
    # 4. SVM
    # 5. KNN

    # import ipdb; ipdb.set_trace()
    # algos = [DecisionTreeClassifier(criterion='entropy', min_samples_split=8, min_samples_leaf=4, max_depth=5),
    #          AdaBoostClassifier(n_estimators=500), SVC(),
    #          LinearSVC(), KNeighborsClassifier(
    #         n_neighbors=20)]  # , MLPClassifier(solver='lbfgs', alpha=1e-5)]#, ExtraTreesClassifier()]

    # print("Description\n")
    # desc = cancer_data.describe()
    # print(desc)
    #
    # desc.to_html('describe_lung.html')
    # subprocess.call(
    #     'wkhtmltoimage -f png --width 0 describe_lung.html describe_lung.png', shell=True)
    #
    # print("Correlations\n")
    # correlation = cancer_data.corr()
    # print(correlation)
    #
    # correlation.to_html('correlation_lung.html')
    # subprocess.call(
    #     'wkhtmltoimage -f png --width 0 correlation_lung.html correlation_lung.png', shell=True)
    #
    # print("Overall\n")

    # Decision Tree
    # accuracies = []
    # for i in range(1, 9):
    #     if i == 1:
    #         accuracy, _, _ = run_supervised_algo_single(cancer_data, [1],
    #                                                     DecisionTreeClassifier(criterion='entropy', min_samples_split=i + 1,
    #                                                                            min_samples_leaf=i,
    #                                                                            max_depth=10))
    #     else:
    #         accuracy, _, _ = run_supervised_algo_single(cancer_data, [1], DecisionTreeClassifier(criterion='entropy', min_samples_split=i, min_samples_leaf=int(i/2), max_depth=5))
    #     accuracies.append((accuracy, i, i/2))
    #
    # print("done", "\n", max(accuracies), "\n\n")

    # accuracy, _, _ = run_supervised_algo_single(cancer_data, [1],
    #                                             DecisionTreeClassifier(criterion='entropy', min_samples_split=3,
    #                                                                    min_samples_leaf=1, max_depth=5))

    # import ipdb; ipdb.set_trace()


    # Adaboost
    # accuracies = []
    # for i in range(1, 10):
    #     for j in np.linspace(0.01, 1, 10):
    #         accuracy, _, _ = run_supervised_algo_single(cancer_data, [1],
    #                                                     AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', min_samples_split=3, min_samples_leaf=1,
    #                                                                            max_depth=10), n_estimators=500, learning_rate=j, random_state=1))
    #         accuracies.append((accuracy, i, j))
    #
    # print("done", "\n", max(accuracies), "\n\n")

    # run_supervised_algo_single(cancer_data, [1],
    #                            AdaBoostClassifier(
    #                                base_estimator=DecisionTreeClassifier(criterion='entropy', min_samples_split=3,
    #                                                                      min_samples_leaf=1,
    #                                                                      max_depth=10), n_estimators=500,
    #                                learning_rate=0.34, random_state=1))
    #
    # import ipdb;
    # ipdb.set_trace()

    # SVM
    # accuracies = []
    # for i in range(1, 10):
    #     accuracy, _, _ = run_supervised_algo_single(cancer_data, [1],
    #                                                 SVC(random_state=1))
    #     accuracies.append((accuracy, i))
    #
    # print("done", "\n", max(accuracies), "\n\n")

    # run_supervised_algo_single(cancer_data, [1],
    #                            SVC(random_state=1))
    #
    # import ipdb; ipdb.set_trace()

    # run_supervised_algo_single(cancer_data, [1],
    #                            SVC())

    # KNN
    # accuracies = []
    # for i in range(1, 10):
    #         accuracy, _, _ = run_supervised_algo_single(cancer_data, [1],
    #                                                 KNeighborsClassifier(n_neighbors=i))
    #         accuracies.append((accuracy, i))
    #
    # print("done", "\n", max(accuracies), "\n\n")

    accuracy, _, _ = run_supervised_algo_single(cancer_data, [1],
                                                KNeighborsClassifier(n_neighbors=1))

    import ipdb; ipdb.set_trace()

    # run_supervised_algo_single(cancer_data, [1], KNeighborsClassifier(n_neighbors=6))

    # for algo in algos:
    #     accuracies = []
    #     for i in range(20):
    #         # accuracy, _, _ = run_supervised_algo_single(contraceptive_data, ['Contraceptive method used_1', 'Contraceptive method used_2',
    #         #                                                                  'Contraceptive method used_3'], algo)
    #         # import ipdb; ipdb.set_trace()
    #         accuracy, _, _ = run_supervised_algo_single(cancer_data, [1], algo)
    #         accuracies.append(accuracy)
    #     print(algo, "\n", s.mean(accuracies), "\n\n")

    # Neural Net

    # accuracies_layers = []
    # accuracies = []
    # for hidden_layers in range(1, 20):
    #     accuracy, _, _ = run_supervised_algo_single(cancer_data, [1],
    #                                                 MLPClassifier(solver='lbfgs', alpha=1e-5,
    #                                                               hidden_layer_sizes=hidden_layers))
    #     accuracies.append(accuracy)
    #     accuracies_layers.append((accuracy, hidden_layers))
    # # run_supervised_algo_single(cancer_data, 1, MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=4))
    #
    # accuracy, depth = max(accuracies_layers, key=itemgetter(0))
    # print("Neural Net\n", accuracy, " Num Hidden Nodes: ", depth, "\n\n")

    # run_supervised_algo_single(cancer_data, 1, MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=13))

    # import ipdb; ipdb.set_trace()

    # print("Feature Selection\n")
    #
    # for algo in algos:
    #     print(algo, "\n\n")
    #     run_supervised_algo_multi(math_data, 'G3', algo)
    #     print("\n\n")

    # call supervised learning algos here
    # run_supervised_algo_multi(math_data, 'G3', LinearRegression())
    # run_supervised_algo_multi(math_data, 'G3', DecisionTreeClassifier(), as_int=True)
    # run_supervised_algo_multi(math_data, 'G3', RandomForestClassifier(n_estimators=10), as_int=True)
    # run_supervised_algo_multi(math_data, 'G3', AdaBoostClassifier(), as_int=True)
    # run_supervised_algo_multi(math_data, 'G3', GradientBoostingClassifier(), as_int=True)
    # accuracy = run_supervised_algo_single(math_data, 'G3', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 63), random_state=1))
    # print(accuracy)


if __name__ == "__main__":
    main()
