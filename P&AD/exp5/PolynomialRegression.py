

# Import required libraries
from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
#%%
#generate dataset for regression
X,y = make_regression(n_samples=100, n_features =5 , noise = 1)
y = pd.Series(y, name = 'output')
X = pd.DataFrame(X)
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
#%%
# plot generated data
plt.hist(X)
plt.show()
#%%
plt.scatter(X.iloc[:,0],y)
plt.show()
plt.scatter(X.iloc[:,1],y)
plt.show()
plt.scatter(X.iloc[:,2],y)
plt.show()
plt.scatter(X.iloc[:,3],y)
plt.show()
plt.scatter(X.iloc[:,4],y)
plt.show()
#%%
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly = PolynomialFeatures(degree = 3)

X_poly = poly.fit_transform(X)
poly.fit(X_poly, y)
lin = LinearRegression()
lin.fit(X_poly, y)

#%%
plt.scatter(X.iloc[:,0], y, color = 'blue')
  
plt.plot(X.iloc[:,0], lin.predict(poly.fit_transform(X)), color = 'red')
plt.title('Polynomial Regression')
plt.show()

#%%
rid=Ridge()
model=rid.fit(X,y,sample_weight=None)
plt.figure(figsize=(15, 15))
plt.plot(X.iloc[:,0],model.predict(X),color="red")
plt.show()
plt.plot(X.iloc[:,1],model.predict(X),color="red")
plt.show()
plt.plot(X.iloc[:,2],model.predict(X),color="red")
plt.show()
plt.plot(X.iloc[:,3],model.predict(X),color="red")
plt.show()
plt.plot(X.iloc[:,4],model.predict(X),color="red")
plt.show()
















