# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 23:38:57 2024

@author: dianel
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, BayesianRidge 
import pandas as pd
from sklearn.metrics import *
# 读取数据


data = pd.read_csv(r"C:\\Users\\dianel\\Downloads\\data\\data.csv")


data['Label'] =data['Z_Scratch']
X = data[['Pixels_Areas', 'Length_of_Conveyer']]
y = data['Label'].astype(int)


#%% no1
model = LogisticRegression().fit(X, y)
pred = model.predict(X)


print(classification_report(y, pred, zero_division=1))
print('accuracy:', accuracy_score(y, pred))
print('loss: ', mean_squared_error(y, pred))
#print('precision:', precision_score(y, pred))
#print('recall:' , recall_score(y, pred))
#print('f1-score:', f1_score(y, pred))
#print('\n')

#%% no2
model = BayesianRidge().fit(X, y)
pred = model.predict(X)


print(classification_report(y, pred, zero_division=1))
print('accuracy:', accuracy_score(y, pred))
print('loss: ', mean_squared_error(y, pred))
print('precision:', precision_score(y, pred))
print('recall:' , recall_score(y, pred))
print('f1-score:', f1_score(y, pred))
print('\n')
