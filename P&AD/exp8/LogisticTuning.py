import numpy as np
from sklearn.datasets import make_classification
from sklearn import linear_model
#from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
#%%
# Generating Data
X, y = make_classification(n_samples = 1000, n_features = 5, n_classes = 2)
x_train = X
y_train = y
#%%
NUM_TRIALS = 30

tuned_parameters = [{ 'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
 'C' : np.logspace(-4, 4, 20),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga']
   }]

score = 'accuracy'
non_nested_scores = np.zeros(NUM_TRIALS)
nested_scores = np.zeros(NUM_TRIALS)
# Loop for each trial
for i in range(NUM_TRIALS):

    # model= GridSearchCV(linear_model.LogisticRegression(), tuned_parameters, scoring= score)
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    model= GridSearchCV(estimator = linear_model.LogisticRegression(), param_grid = tuned_parameters, scoring = score)
    model.fit(x_train, y_train)
    print(model.best_params_)
    non_nested_scores[i] = model.best_score_
    
    
    # Nested CV with parameter optimization
    model = GridSearchCV(estimator= linear_model.LogisticRegression(), param_grid = tuned_parameters, cv=inner_cv, scoring= score)
    nested_score = cross_val_score(model, X=x_train, y=y_train, cv=outer_cv)
    nested_scores[i] = nested_score.mean()
score_difference = non_nested_scores - nested_scores
print("Average difference of {:6f} with std. dev. of {:6f}.".format(score_difference.mean(), score_difference.std()))

#%%
