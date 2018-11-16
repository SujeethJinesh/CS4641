from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

from clustering_algorithms import run_clustering_algo_single
from data_processing import getCleanData

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

    label_cols = ['Class']
    X = np.array(cancer_data.drop(label_cols, 1))  # X is all of our features
    X = preprocessing.scale(X.astype(float)).astype(float)  # scale our features
    num_classes = len(X[0])

    y = np.array(cancer_data[label_cols])  # our y values

    # CheckList
    # 1. K means
    # 2. GMM with EM
    # 3. PCA

    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=4)

    # Kmeans
    confidences = []
    test_accuracies = []
    for i in range(1, num_classes + 1):
        # print(i)
        confidence, test_accuracy = run_clustering_algo_single(X, y, KMeans(n_clusters=i, random_state=0), classifier, neighbors=i)
        confidences.append(confidence)
        test_accuracies.append(test_accuracy)
    print(max(confidences))
    print(max(test_accuracies))

    # GMM with EM
    confidences = []
    test_accuracies = []
    for i in range(1, num_classes + 1):
        # print(i)
        confidence, test_accuracy = run_clustering_algo_single(X, y, GaussianMixture(n_components=i), classifier, neighbors=i)
        confidences.append(confidence)
        test_accuracies.append(test_accuracy)
    print(max(confidences))
    print(max(test_accuracies))

    # PCA
    confidences = []
    test_accuracies = []
    for i in range(1, num_classes + 1):
        # print(i)
        confidence, test_accuracy = run_clustering_algo_single(X, y, PCA(n_components=i), classifier, neighbors=i)
        confidences.append(confidence)
        test_accuracies.append(test_accuracy)
    print(max(confidences))
    print(max(test_accuracies))


if __name__ == "__main__":
    main()
