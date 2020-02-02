from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import datetime
import math
import pandas as pd
from sklearn.impute import SimpleImputer

class FeatureExtractor(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X_df, y):    
        self.imputer = SimpleImputer(strategy='median')
        X_new = X_df.loc[:,['PCT_LACCESS_POP','GROCPTH' ,'SUPERCPTH' ,'CONVSPTH', 'SPECSPTH' ,'PCT_SNAP' ,'PCT_WIC','FFRPTH' ,'FSRPTH','PC_FSRSALES']]
        self.imputer.fit(X_new,y)
        return self

    def transform(self, X_df):
        X_new = X_df.loc[:,['FIPS','PCT_LACCESS_POP','GROCPTH' ,'SUPERCPTH' ,'CONVSPTH', 'SPECSPTH' ,'PCT_SNAP' ,'PCT_WIC','FFRPTH' ,'FSRPTH','PC_FSRSALES']]
        X_new = X_new.groupby('FIPS').transform(lambda x: x.fillna(x.median())) #when a value is missing for one year we take the median value for amongst all years
        X_new = self.imputer.transform(X_new)
        return X_new
