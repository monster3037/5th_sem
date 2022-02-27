from sklearn.datasets import make_regression
#from sklearn.preprocessing import StandardScaler
x,y=make_regression(n_samples=1000,n_features=1,noise=5,random_state=42)
import matplotlib.pyplot as plt
#print(plt.hist(x))
X_mean=x.mean()
Y_mean=y.mean()
c=0
d=0
m=0
n=0
p=0
q=0
from sklearn.model_selection import train_test_split
X_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
X_tmean=X_train.mean()
Y_tmean=y_train.mean()
for i in range(len(X_train)):
        d=X_train[i]-X_tmean
        c=(X_train[i]-X_tmean)**2
        p=y_train[i]-Y_tmean
        m+=d
        n+=c
        q+=p
b1=(m*q)/n
b0=Y_tmean-(b1*X_tmean)
print(b0)
print(b1)
y_pred=[]
for i in range(len(x_test)):
    yi=b0+(b1*x_test[i])
    y_pred.append(yi)
print(y_pred[0:20])
print(y_test[:20])
               
               