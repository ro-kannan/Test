# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 20:51:54 2021

@author: admin
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

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


sns.heatmap(df.corr(),
           annot=True,
           cmap='RdBu')

#without company
#test_train_split
from sklearn.model_selection import train_test_split


df_f = pd.get_dummies(df,drop_first=True)
X= df.drop(['selling_price','company'],axis=1)
X=pd.get_dummies(X,drop_first=True)
y=df.selling_price.values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=42)

#multiple_linear_regression
import statsmodels.api as sm
X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

feat = ['km_driven', 'mileage', 'max_power', 'seats',
       'no_years', 'fuel_Petrol',
       'seller_type_Individual', 'seller_type_Trustmark Dealer',
       'transmission_Manual',
       'owner_Second Owner']

X = X[feat]

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)

X_train = X_train[feat]
X_test = X_test[feat]

from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
 

           
CV = []
R2_train = []
R2_test = []

def car_pred_model(model):
    # Training model
    model.fit(X_train,y_train)
            
    # R2 score of train set
    y_pred_train = model.predict(X_train)
    R2_train_model = r2_score(y_train,y_pred_train)
    R2_train.append(round(R2_train_model,2))
    
    # R2 score of test set
    y_pred_test = model.predict(X_test)
    R2_test_model = r2_score(y_test,y_pred_test)
    R2_test.append(round(R2_test_model,2))
    
    # R2 mean of train set using Cross validation
    cross_val = cross_val_score(model ,X_train ,y_train ,cv=5)
    cv_mean = cross_val.mean()
    CV.append(round(cv_mean,2))
    
    # Printing results
    print("Train R2-score :",round(R2_train_model,2))
    print("Test R2-score :",round(R2_test_model,2))
    print("Train CV scores :",cross_val)
    print("Train CV mean :",round(cv_mean,2))
    
    # Plotting Graphs 
    # Residual Plot of train data
    fig, ax = plt.subplots(1,2,figsize = (10,4))
    ax[0].set_title('Residual Plot of Train samples')
    sns.distplot((y_train-y_pred_train),hist = False,ax = ax[0])
    ax[0].set_xlabel('y_train - y_pred_train')
    
    # Y_test vs Y_train scatter plot
    ax[1].set_title('y_test vs y_pred_test')
    ax[1].scatter(x = y_test, y = y_pred_test)
    ax[1].set_xlabel('y_test')
    ax[1].set_ylabel('y_pred_test')
    
    
    plt.show()
    
car_pred_model(lm)

# lasso regression 
lm_l = Lasso(alpha=.13)
lm_l.fit(X_train,y_train)
np.mean(cross_val_score(lm_l,X_train,y_train, cv= 3))
    
car_pred_model(lm_l)

#alpha_tunning

alpha = []
error = []

for i in range(1,1000):
    alpha.append(i/1000)
    lml = Lasso(alpha=(i/1000))
    error.append(np.mean(cross_val_score(lml,X_train,y_train, cv= 3)))

    
plt.plot(alpha,error)

err=tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
df_err[df_err.error == max(df_err.error)]






#with_company


df_f = pd.get_dummies(df,drop_first=True)
X=df_f.drop('selling_price',axis=1)
y=df.selling_price.values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)
car_pred_model(lm)

# lasso regression 
lm_l = Lasso(alpha=.13)
lm_l.fit(X_train,y_train)
np.mean(cross_val_score(lm_l,X_train,y_train, cv= 3))
    
car_pred_model(lm_l)

#alpha_tunning

alpha = []
error = []

for i in range(1,1000):
    alpha.append(i/1000)
    lml = Lasso(alpha=(i/1000))
    error.append(np.mean(cross_val_score(lml,X_train,y_train, cv= 3)))
# scoring = 'neg_mean_absolute_error'

    
plt.plot(alpha,error)

err=tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
df_err[df_err.error == max(df_err.error)]

## random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
np.mean(cross_val_score(rf,X_train,y_train, cv= 5))
#,scoring = 'neg_mean_absolute_error'

from sklearn.model_selection import GridSearchCV
parameters = {
              'criterion':('mse','mae'), 
              'max_features':('auto','sqrt','log2')}

gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=3)
gs.fit(X_train,y_train)

gs.best_score_
gs.best_estimator_


car_pred_model(gs.best_estimator_)


tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf_gs = gs.best_estimator_.predict(X_test)
tpred_rf = rf.predict(X_test)




from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,tpred_lm)
mean_squared_error(y_test,tpred_lml)
mean_squared_error(y_test,tpred_rf)










