"""
@author: Dhruv Singhal
"""

#%%
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
#%%

#generate dataset
X,y = make_classification(n_samples=1000,n_classes=2)
X=pd.DataFrame(X)
y=pd.Series(y,name=20)
print("X values are:",X.head())
print("Y values are:",y.head())

#%% Visualization

plt.hist(X)
plt.show()

#%%
sns.distplot(y)
plt.show()

#%%
sns.distplot(X)
plt.show()

#%% feature extractionpreprocessing, correlation matrix
_,graph=plt.subplots(figsize=(15,10))
sns.heatmap(X.corr(),annot=True,ax=graph,square=True)
plt.show()

#%%
df=pd.merge(X,y,right_index=True,left_index=True)
print(df.head())
print(df.corr()[[20]].abs().sort_values(by=20,ascending=False))
#%% columns with high correlation will be dropped 
# creating useful data
datasetX=df[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]
datasetY=df[20]
print(datasetY.head())

#%%
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.15,random_state=42)

model=LogisticRegression()
model.fit(X_train,Y_train)
print(model.classes_)

Y_pred=model.predict(X_test)
print(Y_pred)

print("train Accuracy:",model.score(X_train,Y_train))
print("test Accuracy:",model.score(X_test,Y_test))

print(confusion_matrix(Y_test,Y_pred))


























#%%