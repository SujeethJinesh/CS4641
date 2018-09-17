from sklearn import svm, tree

from data_processing import getCleanData
from supervised_learning_algorithms import run_supervised_algo_single, run_supervised_algo_multi
from sklearn.svm import SVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from operator import itemgetter
import statistics as s
import pandas as pd


def main():
    # attributes
    attrs = ['Wife\'s age', 'Wife\'s education', 'Husband\'s education', 'Number of children ever born', 'Wife\'s religion',
             'Wife\'s now working?', 'Husband\'s occupation', 'Standard-of-living index', 'Media exposure', 'Contraceptive method used']

    binary_one_hot_map = {
        # "Wife's religion": [0, 1],
        # "Wife's now working?": [0, 1],
        # "Media exposure": [0, 1]
    }

    # might want to consider separating people by age band, might make it easier to identify relationships.
    normalize_list = [
    ]

    multi_one_hot_list = [
        # "Wife's education", "Husband's education", "Husband's occupation", "Standard-of-living index", "Contraceptive method used"
    ]

    contraceptive_path = 'data/contraceptive/cmc.data'

    contraceptive_data = getCleanData(contraceptive_path, attrs, binary_one_hot_map=binary_one_hot_map, normalize_list=normalize_list,
                             multi_one_hot_list=multi_one_hot_list)

    # CheckList
    # 1. Decision Tree with pruning ✔
    # 2. Neural Network ✔
    # 3. Boosted Decision Tree ✔
    # 4. SVM ✔
    # 5. KNN ✔

    # import ipdb; ipdb.set_trace()
    algos = [DecisionTreeClassifier(criterion='entropy'), DecisionTreeClassifier(criterion='entropy', min_samples_split=100, min_samples_leaf=50, max_depth=10),
             AdaBoostClassifier(n_estimators=30), SVC(), KNeighborsClassifier(n_neighbors=20)]  # , MLPClassifier(solver='lbfgs', alpha=1e-5)]#, ExtraTreesClassifier()]

    print("Description\n")
    print(contraceptive_data.describe())

    print("Correlations\n")
    print(contraceptive_data.corr())

    print("Overall\n")

    for algo in algos:
        accuracies = []
        for i in range(20):
            # accuracy, _, _ = run_supervised_algo_single(contraceptive_data, ['Contraceptive method used_1', 'Contraceptive method used_2',
            #                                                                  'Contraceptive method used_3'], algo)
            accuracy, _, _ = run_supervised_algo_single(contraceptive_data, ['Contraceptive method used'], algo)
            accuracies.append(accuracy)
        print(algo, "\n", s.mean(accuracies), "\n\n")

    accuracies_layers = []
    accuracies = []
    for hidden_layers in range(1, 14):
        accuracy, _, _ = run_supervised_algo_single(contraceptive_data, 'Contraceptive method used',
                                                    MLPClassifier(solver='lbfgs', alpha=1e-5,
                                                                  hidden_layer_sizes=hidden_layers))
        accuracies.append(accuracy)
        accuracies_layers.append((accuracy, hidden_layers))
    # import ipdb; ipdb.set_trace()
    accuracy, depth = max(accuracies_layers, key=itemgetter(1))
    print("Neural Net\n", accuracy, " Num Hidden Nodes: ", depth, "\n\n")

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
