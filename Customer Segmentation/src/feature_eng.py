from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df2 = X.copy()
        # apply log transformation to remove skeyness
        df2['Quantity'] = np.log1p(df2['Quantity'])
        df2['TotalOrders'] = np.log1p(df2['TotalOrders'])
        df2["OrderValue"] = np.log1p(df2['OrderValue'])
        df2["PreviousPurchaseCount"] = np.log1p(df2["PreviousPurchaseCount"])
        return df2