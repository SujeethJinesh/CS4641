from operator import itemgetter

from genetic_selection import GeneticSelectionCV
from sklearn import svm, tree
from sklearn.model_selection import StratifiedKFold

from data_processing import getCleanData
from supervised_learning_algorithms import run_supervised_algo_single, run_supervised_algo_multi
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

import statistics as s
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import numpy as np
from evolutionary_search import EvolutionaryAlgorithmSearchCV

import sys
from IPython.core import ultratb

sys.excepthook = ultratb.FormattedTB(mode='Verbose',
                                     color_scheme='Linux', call_pdb=1)


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

    # cancer_data = getCleanData(cancer_path, attrs, binary_one_hot_map=binary_one_hot_map, normalize_list=normalize_list,
    #                            multi_one_hot_list=multi_one_hot_list, missing_data_marker='?', to_impute=False,
    #                            cols_to_drop=['Sample code number'], normalizer_type='min_max')

    cancer_data = getCleanData(cancer_path, attrs, binary_one_hot_map=binary_one_hot_map,
                               multi_one_hot_list=multi_one_hot_list, missing_data_marker='?', to_impute=False,
                               cols_to_drop=['Sample code number'], normalizer_type='min_max')

    cancer_data.to_csv("breast_cancer_wisconsin.data", index=False)

    import ipdb; ipdb.set_trace()

    # CheckList
    # 1. Neural Network (plain)
    # 2. Neural Network with randomized hill climbing
    # 3. Neural Network with simulated annealing
    # 4. Neural Network with genetic algorithm

    print("Description\n")
    desc = cancer_data.describe()
    print(desc)

    desc.to_html('html/describe.html')
    subprocess.call(
        'wkhtmltoimage -f png --width 0 html/describe.html images/describe.png', shell=True)

    print("Correlations\n")
    correlation = cancer_data.corr()
    print(correlation)

    correlation.to_html('html/correlation.html')
    subprocess.call(
        'wkhtmltoimage -f png --width 0 html/correlation.html images/correlation.png', shell=True)

    print("Overall\n")

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=10)

    # 1. Neural Net (plain)
    # print("Neural Net Plain")
    # print(run_supervised_algo_single(cancer_data, 'Class', clf)[0])

    # 2. Neural Network with randomized hill climbing
    # print("Neural Net with randomized hill climbing")
    # print(run_supervised_algo_single(cancer_data, 'Class', clf)[0])
    #
    # # 3. Neural Network with simulated annealing
    # print("Neural Net with simulated annealing")
    # print(run_supervised_algo_single(cancer_data, 'Class', clf)[0])

    # 4. Neural Network with genetic algorithm
    print("Neural Net with genetic algorithm")
    # paramgrid = {"kernel": ["rbf"],
    #              "C": np.logspace(-9, 9, num=25, base=10),
    #              "gamma": np.logspace(-9, 9, num=25, base=10)}
    # optimizer = EvolutionaryAlgorithmSearchCV(estimator=SVC(),
    #                                           scoring="accuracy",
    #                                           params=paramgrid,
    #                                           cv=StratifiedKFold(n_splits=4),
    #                                           verbose=1,
    #                                           population_size=50,
    #                                           gene_mutation_prob=0.10,
    #                                           gene_crossover_prob=0.5,
    #                                           tournament_size=3,
    #                                           generations_number=5,
    #                                           n_jobs=4)
    optimizer = GeneticSelectionCV(clf,
                                   cv=5,
                                   verbose=1,
                                   scoring="accuracy",
                                   n_population=50,
                                   crossover_proba=0.5,
                                   mutation_proba=0.2,
                                   n_generations=40,
                                   crossover_independent_proba=0.5,
                                   mutation_independent_proba=0.05,
                                   tournament_size=3,
                                   caching=True,
                                   n_jobs=-1)
    print(run_supervised_algo_single(cancer_data, 'Class', clf,
                                     optimizer=optimizer)[0])


if __name__ == "__main__":
    main()
