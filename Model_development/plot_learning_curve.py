## learning curve plotting based on sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, cross_val_predict, KFold, ShuffleSplit

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,exploit_incremental_learning=True,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    
    fig, ax = plt.subplots()
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("# of Training examples",fontsize=14)
    ax.set_ylabel("MAE (eV/atom)",fontsize=14)

    train_sizes, train_scores, test_scores, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,scoring='neg_mean_absolute_error',
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = -np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = -np.std(test_scores, axis=1)
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="#ff7b1d")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="#1d70ff")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="#ff7b1d",
                 label="Training")
    ax.plot(train_sizes, test_scores_mean, 's-', color="#1d70ff",
                 label="Testing")
    ax.legend(loc="best", fontsize=14)
    ax.set_xticks([300, 600, 900, 1200, 1500])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.set_size_inches(3.5,3)
    plt.savefig('learning_curve.png',bbox_inches='tight',transparent=True, dpi=1000)

    return plt
