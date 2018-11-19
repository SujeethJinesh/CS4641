from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

from clustering_algorithms import run_clustering_algo_single, run_Kmeans, run_GMM
from data_processing import getCleanData

import numpy as np
import pandas as pd

from graphing import plot_cross_section


def experiment_1(num_classes_breast_cancer, breast_cancer_X, breast_cancer_Y, num_classes_user, user_knowledge_X,
                 user_knowledge_y):
    # Kmeans Breast Cancer
    confidences = []
    for i in range(1, num_classes_breast_cancer + 1):
        confidence, _ = run_Kmeans(breast_cancer_X, breast_cancer_Y, 1, neighbors=i)
        confidences.append(confidence)
    print(max(confidences))

    # Kmeans User Knowledge
    confidences = []
    for i in range(1, num_classes_user + 1):
        confidence, _ = run_Kmeans(user_knowledge_X, user_knowledge_y, 1, neighbors=i)
        confidences.append(confidence)
    print(max(confidences))

    # GMM with EM Breast Cancer
    confidences = []
    for i in range(1, num_classes_user + 1):
        confidence, _ = run_GMM(breast_cancer_X, breast_cancer_Y, 1, neighbors=i)
        confidences.append(confidence)
    print(max(confidences))

    # GMM with EM User Knowledge
    confidences = []
    for i in range(1, num_classes_user + 1):
        confidence, _ = run_GMM(user_knowledge_X, user_knowledge_y, 1, neighbors=i)
        confidences.append(confidence)
    print(max(confidences))


def experiment_2():
    pass


def experiment_3():
    pass


def experiment_4(classifier, num_classes_breast_cancer, breast_cancer_X, breast_cancer_Y):
    # PCA
    confidences = []
    test_accuracies = []
    for i in range(1, num_classes_breast_cancer + 1):
        # print(i)
        confidence, test_accuracy = run_clustering_algo_single(X, y, PCA(n_components=i), classifier, neighbors=i)
        confidences.append(confidence)
        test_accuracies.append(test_accuracy)
    print(max(confidences))
    print(max(test_accuracies))


def experiment_5(classifier, num_classes_breast_cancer, breast_cancer_X, breast_cancer_Y):
    # Kmeans Breast Cancer
    confidences = []
    test_accuracies = []
    for i in range(1, num_classes_breast_cancer + 1):
        # print(i)
        confidence, test_accuracy = run_clustering_algo_single(breast_cancer_X, breast_cancer_Y,
                                                               KMeans(init='random', n_clusters=i, random_state=0),
                                                               classifier,
                                                               neighbors=i, cross_sections=True)
        confidences.append(confidence)
        test_accuracies.append(test_accuracy)
    print(max(confidences))
    print(max(test_accuracies))

    # Kmeans User Knowledge
    confidences = []
    test_accuracies = []
    for i in range(1, num_classes_user + 1):
        # print(i)
        confidence, test_accuracy = run_clustering_algo_single(user_knowledge_X, user_knowledge_y,
                                                               KMeans(init='random', n_clusters=i, random_state=0),
                                                               classifier,
                                                               neighbors=i, cross_sections=True)
        confidences.append(confidence)
        test_accuracies.append(test_accuracy)
    print(max(confidences))
    print(max(test_accuracies))

    # GMM with EM Breast Cancer
    confidences = []
    test_accuracies = []
    for i in range(1, num_classes_breast_cancer + 1):
        # print(i)
        confidence, test_accuracy = run_clustering_algo_single(breast_cancer_X, breast_cancer_Y,
                                                               GaussianMixture(n_components=i), classifier,
                                                               neighbors=i)
        confidences.append(confidence)
        test_accuracies.append(test_accuracy)
    print(max(confidences))
    print(max(test_accuracies))

    # GMM with EM User Knowledge
    confidences = []
    test_accuracies = []
    for i in range(1, num_classes_user + 1):
        # print(i)
        confidence, test_accuracy = run_clustering_algo_single(user_knowledge_X, user_knowledge_y,
                                                               GaussianMixture(n_components=i), classifier,
                                                               neighbors=i)
        confidences.append(confidence)
        test_accuracies.append(test_accuracy)
    print(max(confidences))
    print(max(test_accuracies))


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

    # Breast Cancer Data
    cancer_path = 'data/breast_cancer/breast_cancer_wisconsin.data.txt'

    cancer_data = getCleanData(cancer_path, attrs, binary_one_hot_map=binary_one_hot_map, normalize_list=normalize_list,
                               multi_one_hot_list=multi_one_hot_list, missing_data_marker='?', to_impute=False,
                               cols_to_drop=['Sample code number'], normalizer_type='min_max')
    label_cols = ['Class']
    X = np.array(cancer_data.drop(label_cols, 1))  # X is all of our features
    breast_cancer_X = preprocessing.scale(X.astype(float)).astype(float)  # scale our features
    num_classes_breast_cancer = len(X[0])

    breast_cancer_Y = np.array(cancer_data[label_cols])  # our y values

    # User Knowledge Data
    knowledge_path = 'data/user_knowledge_modeling/Data_User_Modeling.xls'
    user_knowledge_training_df = pd.read_excel(knowledge_path, sheet_name="Training_Data")
    user_knowledge_training = user_knowledge_training_df.drop(user_knowledge_training_df.columns[list(range(6, 9))],
                                                              axis=1)

    user_knowledge_testing_df = pd.read_excel(knowledge_path, sheet_name="Test_Data")
    user_knowledge_testing = user_knowledge_testing_df.drop(user_knowledge_testing_df.columns[list(range(6, 9))],
                                                            axis=1)

    user_knowledge_merged = pd.merge(user_knowledge_training, user_knowledge_testing,
                                     on=list(user_knowledge_training.columns), how='outer')

    label_cols = [' UNS']
    user_knowledge_X = np.array(user_knowledge_merged.drop(label_cols, 1))  # X is all of our features
    num_classes_user = len(user_knowledge_X[0])

    user_knowledge_y = np.array(user_knowledge_merged[label_cols])  # our y values

    # CheckList
    # 1. K means
    # 2. GMM with EM
    # 3. PCA
    experiment_1
    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=4)

    cross_sections = [["Bare Nuclei", "Mitoses"],
                      ["Uniformity of Cell Shape", "Uniformity of Cell Size"]]
    for cross_section in cross_sections:
        cross_section[0] = cancer_data.columns.get_loc(cross_section[0])
        cross_section[1] = cancer_data.columns.get_loc(cross_section[1])
        plot_cross_section(X, cross_section, "Pre run", "all")

    # import ipdb;
    # ipdb.set_trace()

    experiment_1(num_classes_breast_cancer, breast_cancer_X, breast_cancer_Y, num_classes_user, user_knowledge_X,
                 user_knowledge_y)
    experiment_2()
    experiment_3()
    experiment_4(classifier, num_classes_breast_cancer, breast_cancer_X, breast_cancer_Y)
    experiment_5(classifier, num_classes_breast_cancer, breast_cancer_X, breast_cancer_Y)


if __name__ == "__main__":
    main()
