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
    attrs = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
             'Marginal Adhesion',
             'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

    binary_one_hot_map = {
        "Class": [2, 4]
    }

    # might want to consider separating people by age band, might make it easier to identify relationships.
    normalize_list = [
        'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
        'Marginal Adhesion',
        'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses'
    ]

    multi_one_hot_list = [
    ]

    cancer_path = 'data/breast_cancer/breast_cancer_wisconsin.data.txt'

    cancer_data = getCleanData(cancer_path, attrs, binary_one_hot_map=binary_one_hot_map, normalize_list=normalize_list,
                               multi_one_hot_list=multi_one_hot_list, missing_data_marker='?', to_impute=False,
                               cols_to_drop=['Sample code number'], normalizer_type='min_max')

    # CheckList
    # 1. Decision Tree with pruning
    # 2. Neural Network
    # 3. Boosted Decision Tree
    # 4. SVM
    # 5. KNN

    print("Description\n")
    desc = cancer_data.describe()
    print(desc)

    desc.to_html('describe.html')
    subprocess.call(
        'wkhtmltoimage -f png --width 0 describe.html describe.png', shell=True)

    print("Correlations\n")
    correlation = cancer_data.corr()
    print(correlation)

    correlation.to_html('correlation.html')
    subprocess.call(
        'wkhtmltoimage -f png --width 0 correlation.html correlation.png', shell=True)

    print("Overall\n")

    # Decision Tree
    # accuracies = []
    # for i in range(1, 10):
    #     # import ipdb; ipdb.set_trace()
    #     if i == 1:
    #         accuracy, _, _ = run_supervised_algo_single(cancer_data, ['Class'],
    #                                                     DecisionTreeClassifier(criterion='entropy', min_samples_split=i + 1,
    #                                                                            min_samples_leaf=i,
    #                                                                            max_depth=10))
    #     else:
    #         accuracy, _, _ = run_supervised_algo_single(cancer_data, ['Class'], DecisionTreeClassifier(criterion='entropy', min_samples_split=i, min_samples_leaf=int(i/2), max_depth=5))
    #     accuracies.append((accuracy, i, i/2))
    #
    # print("done", "\n", max(accuracies), "\n\n")
    #
    # import ipdb; ipdb.set_trace()


    # Adaboost
    # accuracies = []
    # for i in range(1, 10):
    #     for j in np.linspace(0.01, 1, 10):
    #         accuracy, _, _ = run_supervised_algo_single(cancer_data, ['Class'],
    #                                                     AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', min_samples_split=8, min_samples_leaf=4,
    #                                                                            max_depth=10), n_estimators=500, learning_rate=j, random_state=1))
    #         accuracies.append((accuracy, i, j))
    #
    # print("done", "\n", max(accuracies), "\n\n")

    run_supervised_algo_single(cancer_data, ['Class'],
                               AdaBoostClassifier(
                                   base_estimator=DecisionTreeClassifier(criterion='entropy', min_samples_split=8,
                                                                         min_samples_leaf=4,
                                                                         max_depth=10), n_estimators=500,
                                   learning_rate=0.56, random_state=1))

    # SVM
    # accuracies = []
    # for i in range(1, 10):
    #     accuracy, _, _ = run_supervised_algo_single(cancer_data, ['Class'],
    #                                                 SVC(random_state=1))
    #     accuracies.append((accuracy, i))
    #
    # print("done", "\n", max(accuracies), "\n\n")

    run_supervised_algo_single(cancer_data, ['Class'],
                               SVC())

    # KNN
    # accuracies = []
    # for i in range(1, 11):
    #         accuracy, _, _ = run_supervised_algo_single(cancer_data, ['Class'],
    #                                                 KNeighborsClassifier(n_neighbors=i))
    #         accuracies.append((accuracy, i))
    #
    # print("done", "\n", max(accuracies), "\n\n")

    run_supervised_algo_single(cancer_data, ['Class'], KNeighborsClassifier(n_neighbors=6))

    # for algo in algos:
    #     accuracies = []
    #     for i in range(20):
    #         # accuracy, _, _ = run_supervised_algo_single(contraceptive_data, ['Contraceptive method used_1', 'Contraceptive method used_2',
    #         #                                                                  'Contraceptive method used_3'], algo)
    #         # import ipdb; ipdb.set_trace()
    #         accuracy, _, _ = run_supervised_algo_single(cancer_data, ['Class'], algo)
    #         accuracies.append(accuracy)
    #     print(algo, "\n", s.mean(accuracies), "\n\n")

    # Neural Net

    # accuracies_layers = []
    # accuracies = []
    # for hidden_layers in range(1, 10):
    #     accuracy, _, _ = run_supervised_algo_single(cancer_data, 'Class',
    #                                                 MLPClassifier(solver='lbfgs', alpha=1e-5,
    #                                                               hidden_layer_sizes=hidden_layers))
    #     accuracies.append(accuracy)
    #     accuracies_layers.append((accuracy, hidden_layers))

    run_supervised_algo_single(cancer_data, 'Class', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=4))


if __name__ == "__main__":
    main()
