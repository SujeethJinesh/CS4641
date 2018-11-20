import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
import itertools


def plot(x_axis, y_axis, title=None):
    plt.plot(x_axis, y_axis, 'g^')
    if title:
        plt.title(title)
    plt.show()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    # import ipdb; ipdb.set_trace()
    plt.close()
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(title + ".png")
    return plt


def plot_confusion_matrix(cm, classes, algorithm,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.close()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("images/" + algorithm + "/" + title + ".png")
    return plt


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def plot_cross_section(transformed_X, cross_section, file_loc, neighbors, experiment_number, dataset_name):
    plt.close()
    plt.figure()
    title = file_loc + " (" + str(neighbors) + " neighbors) " + str(cross_section[0]) + " vs " + str(cross_section[1])
    plt.title(title)
    plt.grid()

    x = transformed_X[:, cross_section[0]]
    y = transformed_X[:, cross_section[1]]
    plt.scatter(x, y)
    plt.savefig("images/experiment_" + str(
        experiment_number) + "/" + dataset_name + "/" + file_loc + "/cross sections/" + title + ".png")


def plot_confidences(confidences, y, xlabel, title, file_loc, experiment_number, dataset_name):
    plt.close()
    plt.figure()
    plt.title(title)
    plt.grid()
    plt.ylabel("confidence")
    plt.xlabel(xlabel)

    plt.plot(y, confidences, 'go--')
    plt.savefig("images/experiment_" + str(
        experiment_number) + "/" + dataset_name + "/" + file_loc + "/confidences/" + title + ".png")


def plot_inertia(confidences, y, xlabel, title, file_loc, experiment_number, dataset_name):
    plt.close()
    plt.figure()
    plt.title(title)
    plt.grid()
    plt.ylabel("inertia")
    plt.xlabel(xlabel)

    plt.plot(y, confidences, 'go--')
    plt.savefig("images/experiment_" + str(
        experiment_number) + "/" + dataset_name + "/" + file_loc + "/inertias/" + title + ".png")


def plot_gaussian_popularity(transformed_X_train, xlabel, title, file_loc, experiment_number, dataset_name):
    plt.close()
    plt.figure()
    plt.title(title)
    plt.grid()
    plt.ylabel("# of points")
    plt.xlabel(xlabel)

    unique, counts = np.unique(transformed_X_train, return_counts=True)
    unique = unique.tolist()
    counts = counts.tolist()
    plt.bar(unique, counts)
    plt.savefig("images/experiment_" + str(
        experiment_number) + "/" + dataset_name + "/" + file_loc + "/gaussian_popularity/" + title + " " + str(len(unique)) + " gaussians.png")
