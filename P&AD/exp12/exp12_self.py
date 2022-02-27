import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_classification


# we create 40 separable points
X, y= make_classification(n_samples=500,n_features=2,n_classes=2,n_informative=2,n_redundant=0,n_repeated=0,random_state=50,class_sep=3.5)
# from sklearn.model_selection import train_test_split
# X,X_t,y,y_t=train_test_split(X1,y1,test_size=0.25,random_state=42)
# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel="linear", C=1000)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(
    XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
)
# plot support vectors
ax.scatter(
    clf.support_vectors_[:, 0],
    clf.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)

plt.show()
#%%

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=2)
kf=KFold(n_splits=70)
score=cross_val_score(clf, X, y, cv=kf)
print("Cross Validation Scores are {}".format(score))
print("Average Cross Validation score :{}".format(score.mean()))


#%%
from sklearn.model_selection import GridSearchCV
tuned_parameters = [{'kernel':["linear"], 'C':[100]},
                    {'kernel':["rbf"], 'C':[90]},
                
                    
                    ]
clf=GridSearchCV(svm.SVC(),tuned_parameters,scoring=('accuracy'))
clf.fit(X,y)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Best Score:",clf.best_score_)
z=clf.cv_results_

#%% kfold cross validation with hyperparameter tuning
from sklearn.model_selection import KFold
k = 5
kf = KFold(n_splits=k, random_state=None)
model = GridSearchCV(svm.SVC(),tuned_parameters,scoring=('accuracy'))
 
for train_index , test_index in kf.split(X):
    X_train , X_test = X[train_index,:],X[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
     
    model.fit(X_train,y_train)
    pred_values = model.predict(X_test)
print("Best parameters set found on development set:")
print()
print(model.best_params_)
print()
print("Best Score:",model.best_score_)
z2=model.cv_results_    
   
#%%