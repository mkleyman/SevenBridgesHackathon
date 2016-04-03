from sklearn.ensemble import gradient_boosting, GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, roc_curve
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from KFoldCV import kfoldCV


def evaluate(r, d):
    tp = 0.
    fp = 0.
    tn = 0.
    fn = 0.
    for i in range(r.shape[0]):
        if r[i] == d[i]:
            if d[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if d[i] == 1:
                fn +=1
            else:
                tn +=1
    k = np.array([tp, fp, tn, fn, (tp+tn)/r.shape[0]])
    return k


def clffit(t1, d1, t2, d2, clf):
    clf.fit(t1, d1)
    r = clf.predict(t2)
    return evaluate(r, d2)


def classify(data, labels):
    lrs = [0.1, 0.5, 1]
    estimateNum = [50, 100, 200]
    depths = [3, 4, 5]
    blr = 0
    be = 0
    bd = 0
    ba = -np.inf
    for lr in lrs:
        for en in estimateNum:
            for d in depths:
                clf = GradientBoostingClassifier(learning_rate=lr, n_estimators=en, max_depth=d)
                s = kfoldCV(data, labels, clffit, 4, clf)
                if s[-1] > ba:
                    ba = s[-1]
                    blr = lr
                    be = en
                    bd = d
    X = data.as_matrix()
    y = labels.as_matrix()
    y = y.reshape([y.shape[1]])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)

    clf = GradientBoostingClassifier(learning_rate=blr, n_estimators=be, max_depth=bd)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    au = accuracy_score(y_test, y_pred)
    fpr, tpr, t = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr, label='roc')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.legend(loc=1)
    plt.savefig('roc.png')

    return au, clf.feature_importances_


if __name__ == '__main__':
    df = pd.DataFrame.from_csv('../data/data_.csv')
    labels = pd.DataFrame(data=[np.random.randint(0, 2) for i in range(10)])
    accuracy, feature_importance = classify(df, labels)
    print accuracy
    print feature_importance
