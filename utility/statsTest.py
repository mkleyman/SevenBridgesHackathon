from scipy import stats as stt
import pandas as pd
import numpy as np

def statistics_test(data, labels):
    d = data.as_matrix()
    y = labels.as_matrix()
    y = y.reshape([y.shape[0]])
    in1 = np.where(y < 1e-5)[0]
    in2 = np.where(y > 1-(1e-5))[0]
    l = []
    for i in range(d.shape[1]):
        s, p = stt.ranksums(d[in1, i], d[in2, i])
        l.append(p)
    df = pd.DataFrame(data=np.array(l)*len(l), index=data.columns.values, columns=['pvalue'])
    return df

if __name__ == '__main__':
    df = pd.DataFrame.from_csv('../data/data_.csv')
    labels = pd.DataFrame(data=[np.random.randint(0, 2) for i in range(10)])
    print labels
    d = statistics_test(df, labels)
    print d