# -*- coding: utf-8 -*-
"""

@author: Dhruv Singhal
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
#%%
X,y=make_regression(n_samples=100,n_features=10,n_informative=4,random_state=0)
sns.distplot(y)

#%%
from sklearn.decomposition import PCA
pca=PCA(n_components=5)
principalComponents=pca.fit_transform(X)
X_new=pd.DataFrame(data=principalComponents,columns=('PC1','PC2','PC3','PC4','PC5'))
print(X_new.head())
#%%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train , X_test , Y_train , Y_test = train_test_split(X_new, y, train_size=0.20,random_state=0)
lin=LinearRegression()
model=lin.fit(X_train,Y_train)
y_pred=model.predict(X_test)
#%%
print('Coefficients: ', model.coef_)
print('Variance score: {}'.format(model.score(X_test, Y_test)))
print("Test Score",model.score(X_train,Y_train))
print("Test Score",model.score(X_test,Y_test))
#%%
#training
plt.scatter(model.predict(X_train),model.predict(X_train) - Y_train,color = "green", s = 10, label = 'Train data')
#testing
plt.scatter(model.predict(X_test),model.predict(X_test) - Y_test,color = "orange", s = 10, label = 'Testing data')


#%%
import math
from sklearn.metrics import mean_squared_error
print("RMSE",math.sqrt(mean_squared_error(Y_test,y_pred)))
from sklearn.metrics import r2_score
print("R^2",r2_score(Y_test,y_pred))


#%%
print(np.max(Y_train),np.min(Y_train))

#%%
plt.figure(figsize=(20,10))
plt.plot(y_pred)
plt.plot(Y_test)




#%%
X_train2 , X_test2 , Y_train2 , Y_test2 = train_test_split(X, y, train_size=0.20,random_state=0)
lin=LinearRegression()
model2=lin.fit(X_train2,Y_train2)
y_pred2=model.predict(X_test)
#%%
print('Coefficients: ', model2.coef_)
print('Variance score: {}'.format(model2.score(X_test2, Y_test2)))
print("Test Score",model2.score(X_train2,Y_train2))
print("Test Score",model2.score(X_test2,Y_test2))

#%%
import math
from sklearn.metrics import mean_squared_error
print("RMSE",math.sqrt(mean_squared_error(Y_test2,y_pred2)))
from sklearn.metrics import r2_score
print("R^2",r2_score(Y_test2,y_pred2))


#%%
print(np.max(Y_train2),np.min(Y_train2))

#%%
plt.figure(figsize=(20,10))
plt.plot(y_pred2)
plt.plot(Y_test2)



