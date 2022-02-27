import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import warnings
#%%
warnings.filterwarnings('ignore')
#data
data = pd.read_csv('winequality-red.csv')
data.head()
#%%
data.info()

#%%
data.describe()
#%%
data.corr()
#%%
# eda
import seaborn as sns
sns.heatmap(data.corr(), annot=True)
#%%

X = data.drop("quality", axis=1)
y = data[['quality']]
scaler = StandardScaler()
X = scaler.fit_transform(X)

#%%
print(X)
#%%
print(y.describe())
#%%
#train test split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
#models

linear = LinearRegression()

kNN_model = KNeighborsClassifier(n_neighbors=5)

forest_descision = RandomForestClassifier(max_depth=5, random_state=0)

SVC_model = SVC()

model_list = [linear,kNN_model,forest_descision,SVC_model]
#%%
linear.fit(X_train, y_train)
print("linear regression: ", linear.score(X_test, y_test))

kNN_model.fit(X_train, y_train)
print("knn: ", kNN_model.score(X_test, y_test))

forest_descision.fit(X_train, y_train)
print("random forest: ", forest_descision.score(X_test, y_test))

SVC_model.fit(X_train, y_train)
#%%
linear = LinearRegression()

kNN_model = KNeighborsClassifier(n_neighbors=50)

forest_descision = RandomForestClassifier(max_depth=1, random_state=2)

SVC_model = SVC(kernel='linear')
#%%
linear.fit(X_train, y_train)
print("linear regression: ", linear.score(X_test, y_test))

kNN_model.fit(X_train, y_train)
print("knn: ", kNN_model.score(X_test, y_test))

forest_descision.fit(X_train, y_train)
print("random forest: ", forest_descision.score(X_test, y_test))

SVC_model.fit(X_train, y_train)
print("SVM: ", SVC_model.score(X_test, y_test))