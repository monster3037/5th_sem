import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import seaborn as sns


#%%
#Reading data
data=pd.read_csv("AirQualityUCI.csv")

print(data.info())

print(data.describe())

print(data.isnull().sum())

#%%
print(data.corr())

#%%
sns.heatmap(data.corr(), annot=True)
#%%
col_=data.columns.tolist()[2:]
for i in data.columns.tolist()[2:]:
    sns.lmplot(x=i,y='RH',data=data,markers='.')
#%%
from sklearn.preprocessing import StandardScaler         #import normalisation package
from sklearn.model_selection import train_test_split      #import train test split
from sklearn.linear_model import LinearRegression         #import linear regression package
from sklearn.metrics import mean_squared_error,mean_absolute_error   #import mean squared error and mean absolute error
#%%
X=data[col_].drop('RH',1)     #X-input features
y=data['RH']

#%%
ss=StandardScaler()     #initiatilise
X_std=ss.fit_transform(X)     #apply stardardisation
#%%

X_train, X_test, y_train, y_test=train_test_split(X_std,y,test_size=0.3, random_state=42)
#%%
print('Training data size:',X_train.shape)
print('Test data size:',X_test.shape)

#%%
lr=LinearRegression()
lr_model=lr.fit(X_train,y_train)          #fit the linear model on train data
print('Intercept:',lr_model.intercept_)
print('--------------------------------')
print('Slope:')
print(list(zip(X.columns.tolist(),lr_model.coef_)))

#%%
y_pred=lr_model.predict(X_test)                      #predict using the model
rmse=np.sqrt(mean_squared_error(y_test,y_pred))      #calculate rmse
print('RMSE of model:',rmse)
print("linear regression: ", lr_model.score(X_test, y_test))

#%%

from sklearn.ensemble import RandomForestRegressor           #import random forest regressor
rf_reg=RandomForestRegressor()
#%%

rf_model=rf_reg.fit(X_train,y_train)         #fit model   
y_pred_rf=rf_model.predict(X_test)           #predict

#%%
#Calculate RMSE
print('RMSE of predicted RH in RF model:',np.sqrt(mean_squared_error(y_test,y_pred_rf)))

#%%
print("Random Forest: ", rf_model.score(X_test, y_test))