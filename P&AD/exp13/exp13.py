"""
@author: Dhruv Singhal
"""


#%%
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
#%%

#generate dataset
X,y = make_classification(n_samples=1000,n_classes=2,n_features=5)
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
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.15,random_state=42)

model=RandomForestClassifier()
model.fit(X_train,Y_train)
print(model.classes_)

Y_pred=model.predict(X_test)
print(Y_pred)

print("train Accuracy:",model.score(X_train,Y_train))
print("test Accuracy:",model.score(X_test,Y_test))

print(confusion_matrix(Y_test,Y_pred))


























#%%
print("hello")
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
clf = RandomForestClassifier(max_depth=5, random_state=0)
kf=KFold(n_splits=7)
score=cross_val_score(clf, X, y, cv=kf)
print("Cross Validation Scores are {}".format(score))
print("Average Cross Validation score :{}".format(score.mean()))


#%%
from sklearn.model_selection import GridSearchCV
tuned_parameters = [{'n_estimators':[10,20,40,100],'criterion':['gini', 'entropy'],
                     'max_features':['auto', 'sqrt', 'log2'],'bootstrap':[True,False]}]
clf=GridSearchCV(RandomForestClassifier(),tuned_parameters,scoring=('accuracy'))
clf.fit(X,y)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Best Score:",clf.best_score_)
z=clf.cv_results_

#%%

