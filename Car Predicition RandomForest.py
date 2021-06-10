# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 20:14:31 2021

@author: admin
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

df=pd.read_csv('Car details v3.csv')
df=df.dropna()
df.drop(df.index[df['owner']=='Test Drive Car'],inplace=True)
df[['seats']] = df[['seats']].fillna(0.0).astype(int)
df['mileage']=df['mileage'].str.replace(' kmpl','')
df['mileage']=df['mileage'].str[0:5].astype(float)
df['engine']=df['engine'].str.replace('CC','').astype(int)
df['no_years'] = 2021
df['no_years'] = 2021 - df['year']
df['max_power']=df['max_power'].str.replace('bhp','').astype(float)
df.drop(['torque','year'],axis=1,inplace=True)
df['company'] = df['name'].str.split(expand=True)[0].astype('str')
df.loc[df['company'] == 'Ashok','company'] = 'Ashok Leyland'
df.loc[df['company'] == 'Land','company'] = 'Land Rover'
df.drop('name',axis=1,inplace=True)
df['selling_price'] = df['selling_price']/100000
df['km_driven'] = df['km_driven']/10000

from sklearn.model_selection import train_test_split


df_f = pd.get_dummies(df,drop_first=True)
X= df.drop(['selling_price','company'],axis=1)
X=pd.get_dummies(X,drop_first=True)
y=df.selling_price.values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train,y_train)

from sklearn.model_selection import cross_val_score
y_pred_train = rf.predict(X_train)
R2_train_model = r2_score(y_train,y_pred_train)
y_pred_test = rf.predict(X_test)
R2_test_model = r2_score(y_test,y_pred_test)
rmse=mean_squared_error(y_test,y_pred_test,squared=False)
cross_val = cross_val_score(rf ,X_train ,y_train ,cv=5)

results=pd.DataFrame({'Model': 'RandomForest',
                      'R Squared': R2_test_model,
                      'CV score mean': [np.mean(cross_val)],
                      'RMSE':rmse})
sns.distplot(y_test-y_pred_test)
sns.scatterplot(x=y_pred_test,y=y_test)


import pickle

file = open('random_forest_regression.pkl','wb')
pickle.dump(rf,file)