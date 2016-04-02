from sklearn.cross_validation import KFold
import numpy as np
import pandas as pd


def kfoldCV(X, y, f, K):
    X = X.as_matrix()
    y = y.as_matrix()
    kf = KFold(X.shape[0], n_folds=K)
    s = np.zeros([1, 4])
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        s += f(pd.DataFrame(data=X_train), pd.DataFrame(data=y_train), pd.DataFrame(data=X_test),
               pd.DataFrame(data=y_test))
    return s / K
