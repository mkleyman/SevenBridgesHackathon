from sklearn.cross_validation import KFold
import numpy as np
import pandas as pd


def kfoldCV(X, y, f, K, clf=None):
    X = X.as_matrix()
    y = y.as_matrix()
    y = y.reshape([y.shape[1]])
    kf = KFold(X.shape[0], n_folds=K, random_state=0)
    s = np.zeros([5,])
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        s += f(X_train, y_train, X_test, y_test, clf=clf)
    return s / K
