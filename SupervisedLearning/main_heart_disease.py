from sklearn import svm, tree

from data_processing import getCleanData
from supervised_learning_algorithms import run_supervised_algo_single, run_supervised_algo_multi
from sklearn.svm import SVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

import statistics as s
import pandas as pd


def main():
    # attributes
    attrs = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob',
             'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup',
             'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel',
             'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']

    binary_one_hot_map = {
        'school': ["GP", "MS"],
        'sex': ["F", "M"],
        'address': ["U", "R"],
        'famsize': ["LE3", "GT3"],
        'Pstatus': ["T", "A"],
        'schoolsup': ["no", "yes"],
        'famsup': ["no", "yes"],
        'paid': ["no", "yes"],
        'activities': ["no", "yes"],
        'nursery': ["no", "yes"],
        'higher': ["no", "yes"],
        'internet': ["no", "yes"],
        'romantic': ["no", "yes"],
    }

    # might want to consider separating people by age band, might make it easier to identify relationships.
    normalize_list = [
        'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'G3', 'absences'
    ]

    multi_one_hot_list = [
        'age', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures'
    ]

    math_path = 'data/student/student-mat.csv'
    por_path = 'data/student/student-por.csv'

    math_data = getCleanData(math_path, attrs, binary_one_hot_map=binary_one_hot_map, normalize_list=normalize_list,
                             multi_one_hot_list=multi_one_hot_list, row_num_to_drop=0, cols_to_drop=['G1', 'G2'])
    por_data = getCleanData(por_path, attrs, binary_one_hot_map=binary_one_hot_map, normalize_list=normalize_list,
                            multi_one_hot_list=multi_one_hot_list, row_num_to_drop=0, cols_to_drop=['G1', 'G2'])

    # CheckList
    # 1. Decision Tree with pruning ✔
    # 2. Neural Network ✔
    # 3. Boosted Decision Tree ✔
    # 4. SVM ✔
    # 5. KNN ✔

    import ipdb; ipdb.set_trace()
    algos = [DecisionTreeClassifier(), AdaBoostClassifier(n_estimators=30), SVC(),
             KNeighborsClassifier(
                 n_neighbors=20)]  # , MLPClassifier(solver='lbfgs', alpha=1e-5)]#, ExtraTreesClassifier()]

    print("Overall\n")

    for algo in algos:
        accuracies = []
        for i in range(20):
            accuracy, _, _ = run_supervised_algo_single(por_data, 'G3', algo)
            accuracies.append(accuracy)
        print(algo, "\n", s.mean(accuracies), "\n\n")

    accuracies = []
    for hidden_layers in range(1, 300):
        accuracy, _, _ = run_supervised_algo_single(por_data, 'G3',
                                                    MLPClassifier(solver='lbfgs', alpha=1e-5,
                                                                  hidden_layer_sizes=hidden_layers))
        accuracies.append((accuracy, hidden_layers))
    accuracy, depth = s.mean(accuracies)
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
