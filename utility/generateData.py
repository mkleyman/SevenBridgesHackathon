import pandas as pd
import numpy as np

for i in range(10):
    col = []
    for j in range(10):
        if j not in [2, 3, 5]:
            if np.random.random() > 0.4:
                col.append('gene'+str(j))
        else:
            col.append('gene'+str(j))
    ind = ['ind'+str(i)]
    data = np.random.random([1, len(col)])
    df = pd.DataFrame(data=data, index=ind, columns=col)
    df.to_csv('../data/data_'+str(i)+'.csv')