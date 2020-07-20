# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:20:24 2020

@author: NAIDU
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator

class SelectColulmnTransformer(TransformerMixin,BaseEstimator):
    def __init__(self,columns=None):
        print('init SelectColulmnTransformer')
        self.columns = columns
        
    def transform(self,X,**param):
        print('transform called    param {}'.format( param))
        cpy_df = X[self.columns].copy()
        cpy_df = cpy_df.apply(lambda x : [i.upper() for i in x])
        return cpy_df
    
    def fit(self,X,y=None,**param):
        print('fit called  with pram {}'.format( param))
        return self
    

df = pd.DataFrame({'A':['a','b']})
pipe = Pipeline([('columntransform',SelectColulmnTransformer(['A']))])
newdf = pipe.fit(df,None)
newdf = pipe.transform(df)
print(newdf)

