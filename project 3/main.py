from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from clustering_algorithms import run_clustering_algo_single, run_Kmeans, run_GMM, run_PCA, run_Neural_Net
from data_processing import getCleanData

import numpy as np
import pandas as pd

from graphing import plot_confidences, plot_inertia, plot_gaussian_popularity, plot_components


def experiment_1(num_classes_breast_cancer, breast_cancer_data, num_classes_user, user_knowledge_data,
                 experiment_number=1):
    breast_cancer_range = list(range(1, num_classes_breast_cancer + 1))
    user_knowledge_range = list(range(1, num_classes_user + 1))

    breast_cancer_X_train, breast_cancer_X_test, breast_cancer_y_train, breast_cancer_y_test = breast_cancer_data
    user_knowledge_X_train, user_knowledge_X_test, user_knowledge_y_train, user_knowledge_y_test = user_knowledge_data

    # Kmeans Breast Cancer
    confidences = []
    inertias = []
    breast_cancer_kmeans_data = []
    for i in breast_cancer_range:
        confidence, inertia, transformed_X_train, transformed_X_test, df = run_Kmeans(breast_cancer_X_train,
                                                                                      breast_cancer_X_test,
                                                                                      breast_cancer_y_train,
                                                                                      breast_cancer_y_test, 1,
                                                                                      "breast_cancer", neighbors=i)
        breast_cancer_kmeans_data.append((transformed_X_train, transformed_X_test, breast_cancer_y_train,
                                          breast_cancer_y_test))
        confidences.append(confidence)
        inertias.append(inertia)
    plot_confidences(confidences, breast_cancer_range, "neighbors", "Breast Cancer Confidence vs Neighbors (Kmeans)",
                     "Kmeans", experiment_number, "breast_cancer")
    plot_inertia(inertias, breast_cancer_range, "neighbors", "Breast Cancer Inertia vs Neighbors (Kmeans)", "Kmeans",
                 experiment_number,
                 "breast_cancer")
    plot_components(df, "Breast Cancer First and Second Principal Components colored by Class", "Kmeans",
                    experiment_number,
                    "breast_cancer")

    # Kmeans User Knowledge
    confidences = []
    inertias = []
    for i in user_knowledge_range:
        confidence, inertia, transformed_X_train, transformed_X_test, df = run_Kmeans(user_knowledge_X_train,
                                                                                      user_knowledge_X_test,
                                                                                      user_knowledge_y_train,
                                                                                      user_knowledge_y_test,
                                                                                      experiment_number,
                                                                                      "user_knowledge", neighbors=i)
        confidences.append(confidence)
        inertias.append(inertia)
    plot_confidences(confidences, user_knowledge_range, "neighbors", "User Knowledge Confidence vs Neighbors (Kmeans)",
                     "Kmeans", experiment_number, "user_knowledge")
    plot_inertia(inertias, user_knowledge_range, "neighbors", "User Knowledge Inertia vs Neighbors (Kmeans)", "Kmeans",
                 experiment_number, "user_knowledge")
    plot_components(df, "User Knowledge First and Second Principal Components colored by Class", "Kmeans",
                    experiment_number,
                    "user_knowledge")

    # GMM with EM Breast Cancer
    confidences = []
    breast_cancer_gmm_data = []
    for i in breast_cancer_range:
        confidence, transformed_X_train, transformed_X_test = run_GMM(breast_cancer_X_train, breast_cancer_X_test,
                                                                      breast_cancer_y_train, breast_cancer_y_test,
                                                                      neighbors=i)
        breast_cancer_gmm_data.append((transformed_X_train, transformed_X_test, breast_cancer_y_train,
                                       breast_cancer_y_test))
        confidences.append(confidence)
        plot_gaussian_popularity(transformed_X_train, "gaussians", "Breast Cancer Confidence vs Cluster (GMM with EM)",
                                 "GMM", experiment_number, "breast_cancer")
    plot_confidences(confidences, breast_cancer_range, "gaussians",
                     "Breast Cancer Confidence vs Neighbors (GMM with EM)", "GMM", experiment_number, "breast_cancer")

    # GMM with EM User Knowledge
    confidences = []
    for i in user_knowledge_range:
        confidence, transformed_X_train, transformed_X_test = run_GMM(user_knowledge_X_train, user_knowledge_X_test,
                                                                      user_knowledge_y_train, user_knowledge_y_test,
                                                                      neighbors=i)
        confidences.append(confidence)
        plot_gaussian_popularity(transformed_X_train, "gaussians", "User Knowledge Points vs Cluster (GMM with EM)",
                                 "GMM", experiment_number, "user_knowledge")
    plot_confidences(confidences, user_knowledge_range, "gaussians",
                     "User Knowledge Confidence vs Neighbors (GMM with EM)", "GMM", experiment_number, "user_knowledge")

    return breast_cancer_kmeans_data, breast_cancer_gmm_data


def experiment_2(breast_cancer_data, user_knowledge_data):
    breast_cancer_X_train, breast_cancer_X_test, breast_cancer_y_train, breast_cancer_y_test = breast_cancer_data
    user_knowledge_X_train, user_knowledge_X_test, user_knowledge_y_train, user_knowledge_y_test = user_knowledge_data

    # PCA Breast Cancer
    _, breast_cancer_transformed_X_train, breast_cancer_transformed_X_test, df = run_PCA(breast_cancer_X_train,
                                                                                         breast_cancer_X_test,
                                                                                         breast_cancer_y_train,
                                                                                         breast_cancer_y_test)
    plot_components(df, "Breast Cancer First and Second Principal Components colored by Class", "PCA", 2,
                    "breast_cancer")

    # PCA User Knowledge
    _, user_knowledge_transformed_X_train, user_knowledge_transformed_X_test, df = run_PCA(user_knowledge_X_train,
                                                                                           user_knowledge_X_test,
                                                                                           user_knowledge_y_train,
                                                                                           user_knowledge_y_test)
    plot_components(df, "User Knowledge First and Second Principal Components colored by Class", "PCA", 2,
                    "user_knowledge")

    breast_cancer_data = (
        breast_cancer_transformed_X_train, breast_cancer_transformed_X_test, breast_cancer_y_train,
        breast_cancer_y_test)
    user_knowledge_data = (
        user_knowledge_transformed_X_train, user_knowledge_transformed_X_test, user_knowledge_y_train,
        user_knowledge_y_test)

    return breast_cancer_data, user_knowledge_data


def experiment_3(num_classes_breast_cancer, breast_cancer_data, num_classes_user, user_knowledge_data):
    # Post PCA values
    experiment_1(num_classes_breast_cancer, breast_cancer_data, num_classes_user, user_knowledge_data,
                 experiment_number=3)


def experiment_4(breast_cancer_data):
    # PC
    path = "images/experiment_4/breast_cancer/PCA/"
    test_accuracy = run_Neural_Net(breast_cancer_data, path, "PCA")
    print(test_accuracy)


def experiment_5(breast_cancer_kmeans_data, breast_cancer_gmm_data):
    path_kmeans = "images/experiment_5/breast_cancer/Kmeans/"
    path_gmm = "images/experiment_5/breast_cancer/GMM/"

    # Kmeans Breast Cancer
    test_accuracies = []
    for index, breast_cancer_data in enumerate(breast_cancer_kmeans_data):
        test_accuracy = run_Neural_Net(breast_cancer_data, path_kmeans, "Kmeans", neighbors=index + 1)
        test_accuracies.append(test_accuracy)
    print(max(test_accuracies))

    # GMM with EM Breast Cancer
    test_accuracies = []
    for index, breast_cancer_data in enumerate(breast_cancer_gmm_data):
        test_accuracy = run_Neural_Net(breast_cancer_data, path_gmm, "GMM", neighbors=index + 1)
        test_accuracies.append(test_accuracy)
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

    # cross_sections = [["Bare Nuclei", "Mitoses"],
    #                   ["Uniformity of Cell Shape", "Uniformity of Cell Size"]]
    # for cross_section in cross_sections:
    #     cross_section[0] = cancer_data.columns.get_loc(cross_section[0])
    #     cross_section[1] = cancer_data.columns.get_loc(cross_section[1])
    #     plot_cross_section(X, cross_section, "Pre run", "all")

    breast_cancer_X_train, breast_cancer_X_test, breast_cancer_y_train, breast_cancer_y_test = train_test_split(
        breast_cancer_X, breast_cancer_Y, test_size=0.2)  # produces good shuffled train and test sets
    user_knowledge_X_train, user_knowledge_X_test, user_knowledge_y_train, user_knowledge_y_test = train_test_split(
        user_knowledge_X, user_knowledge_y, test_size=0.2)  # produces good shuffled train and test sets

    breast_cancer_data = (breast_cancer_X_train, breast_cancer_X_test, breast_cancer_y_train, breast_cancer_y_test)
    user_knowledge_data = (user_knowledge_X_train, user_knowledge_X_test, user_knowledge_y_train, user_knowledge_y_test)

    # Experiment 1
    breast_cancer_kmeans_data, breast_cancer_gmm_data = experiment_1(num_classes_breast_cancer, breast_cancer_data,
                                                                     num_classes_user, user_knowledge_data)

    # Experiment 2
    pca_breast_cancer_data, pca_user_knowledge_data = experiment_2(breast_cancer_data, user_knowledge_data)

    # Experiment 3
    experiment_3(num_classes_breast_cancer, pca_breast_cancer_data, num_classes_user, pca_user_knowledge_data)

    # Experiment 4
    experiment_4(pca_breast_cancer_data)

    # Experiment 5
    experiment_5(breast_cancer_kmeans_data, breast_cancer_gmm_data)


if __name__ == "__main__":
    main()
