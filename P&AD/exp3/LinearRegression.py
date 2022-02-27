#%%
# import required libraries
from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error ,mean_absolute_error
import numpy as np
import seaborn as sns
from math import sqrt
import warnings
warnings.filterwarnings('ignore')
#%%
#generate dataset for regression
X,y = make_regression(n_samples=1000, n_features = 1, shuffle = True, noise = 5,random_state=86)
#%%
#Statistics
stdX = np.std(X)
meanX = np.mean(X)

stdY = np.std(y)
meanY = np.mean(y)
print(meanX)
print(stdX) 
print(meanY)
print(stdY)
#%%
# plot generated data
plt.hist(X)
plt.show()
#%%
plt.scatter(X,y)
plt.show()
#%%
# As mean of this dataset is already near to 0, so standard scaling is not required
#%%
# Preprocessing  
#%%
plt.hist(X)
plt.show()
#%%
#Plot for data after preprocessing
plt.scatter(X,y)
plt.show()
#%%
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
#%%
sns.distplot(y_train)
plt.show()
#%%
sns.distplot(y_test)
plt.show()
#%%
#Performing Linear Regression || Using Traditional Method
#Initializing variables
n = len(X_train)
num = 0
denom = 0
#%%
#Calculating slope & intercept
for i in range(n):
    num += (X_train[i]-meanX) * (y_train[i]-meanY)
    denom += (X_train[i] - meanX)**2
m = num/denom
c = meanY - (m*meanX)
print(m, ',', c)

min_X = 1 - np.min(X_train)
max_X = 1 - np.max(X_train)
print(min_X, ',', max_X)
#%%
#calculating the values of x & y
x = np.linspace(min_X, max_X,1000)
y = (m*x)+c
#%%  Plot Regression Line
plt.scatter(X_train,y_train,color='b')
plt.plot(x,y,color='r')
plt.title('Simple Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
#%%
#calculating error
y_pred = []
sum_pred = 0
sum_actual = 0
for i in range(len(X_test)):
    val = (m*X_test[i]+c)
    y_pred.append(val)
mse = mean_squared_error(y_test, y_pred)
print("MSE", mse)
r2 = sqrt(mse)
print("R2",r2)
mae = mean_absolute_error(y_test, y_pred)
print("MAE", mae)
#%%
# Perform Linear Regression || Using sklearn
# Build Model
LRmodel = LinearRegression()
LRmodel.fit(X_train, y_train)
#%%
# Error Calculation
print("--------------------------------------")
y_pred1 = LRmodel.predict(X_test)
mse1 = mean_squared_error(y_test, y_pred1) 
print("MSE with sklearn", mse1)
r2new = sqrt(mse1)
print("R2 with sklearn",r2new)
mae1 = mean_absolute_error(y_test, y_pred1)
print("MAE with sklearn", mae1)






#%%
