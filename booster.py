import pandas as pd
import numpy as np
import ggplot
import xgboost

def main(train_genes, train_classes, test_genes, test_classes):
    ''' Expects pandas dataframes '''


# We'll impute missing values using the median for numeric columns and the most
# common value for string columns.
# This is based on some nice code by 'sveitser' at http://stackoverflow.com/a/25562948
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

def train(train_genes, train_classes):


def test(model, test_genes, test_classes):


