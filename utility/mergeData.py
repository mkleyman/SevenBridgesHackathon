import pandas as pd
import numpy as np

def mergeData(l):
    # nameCount = {}
    # for k in l:
    #     cl = list(k.columns.values)
    #     for c in cl:
    #         if c not in nameCount:
    #             nameCount[c] = 1
    #         else:
    #             nameCount[c] += 1
    # fl = []
    # for k in nameCount:
    #     if nameCount[k] == len(l):
    #         fl.append(k)
    # df = pd.DataFrame(columns=fl)
    # indexes = []
    # for i in range(len(l)):
    #     # print l[i][fl].as_matrix()
    #     df.loc[len(df)] = l[i][fl].as_matrix()[0]
    #     indexes.append()
    # df.index.names = indexes

    df = pd.concat(l, axis=0, join='inner', join_axes=None, ignore_index=False,
       keys=None, levels=None, names=None, verify_integrity=False)

    return df


def loadData():
    l = []
    for i in range(10):
        d = pd.DataFrame.from_csv('../data/data_'+str(i)+'.csv')
        l.append(d)
    return l


if __name__ == '__main__':
    l = loadData()
    d = mergeData(l)
    print d