import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%%
dataset=pd.read_csv("B:/3rd year/5th sem/ML Industry/winequalityN.csv")
dataset.head()
dataset.describe()
dataset.info()
dataset.isnull().sum()

#%%

sns.pairplot(dataset)
plt.show()

#%%

dataset.hist(bins=20,figsize=(10,10))# bins is number of parts in which whole info is divided
plt.show()
#%%
plt.figure(figsize=[15,6])
plt.bar(dataset['quality'],dataset['alcohol'])
plt.xlabel('Quality')
plt.ylabel('Alcohol')
plt.show()

#%%
plt.figure(figsize=[15,10])
sns.heatmap(dataset.corr(),annot=True) # If True, write the data value in each cell.
plt.show()
#From this correlation visualization, we will find which features are correlated with other features. 
#so we will use a python program to find those features.
#%%
col=[]
for i in range(len(dataset.corr().keys())):
    for j in range(i):
        if abs(dataset.corr().iloc[i,j])>0.7:
            col=dataset.corr().columns[i]
print(col)

#%%
df=dataset.drop('total sulfur dioxide',axis=1)
#%%  
df.update(df.fillna(df.mean()))
df.isnull().sum()
#%%

category=df.select_dtypes(include='O')
df_dummies=pd.get_dummies(df,drop_first= True)
df_dummies
#%%
df_dummies['best quality']=[1 if x>=7 else 0 for x in dataset.quality]
print(df_dummies)

#%%
from sklearn.model_selection import train_test_split
x=df_dummies.drop(['quality','best quality'],axis=1)
y=df_dummies['best quality']
x_train,y_train,x_test,y_test= train_test_split(x,y,test_size=0.3,random_state=42)
#%%

#importing module
from sklearn.preprocessing import MinMaxScaler
# creating normalization object 
norm = MinMaxScaler()
# fit data
norm_fit = norm.fit(x_train)
new_xtrain = norm_fit.transform(x_train)
new_xtest = norm_fit.transform(x_test)
# display values

#%%
# importing modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
#creating RandomForestClassifier constructor
rnd = RandomForestClassifier()
# fit data
fit_rnd = rnd.fit(new_xtrain,y_train)
# predicting score
rnd_score = rnd.score(new_xtest,y_test)
print('score of model is : ',rnd_score)
# display error rate
print('calculating the error')
# calculating mean squared error
rnd_MSE = mean_squared_error(y_test,y_predict)
# calculating root mean squared error
rnd_RMSE = np.sqrt(MSE)
# display MSE
print('mean squared error is : ',rnd_MSE)
# display RMSE
print('root mean squared error is : ',rnd_RMSE)
print(classification_report(x_predict,y_test))
print(new_xtrain)
#%%







