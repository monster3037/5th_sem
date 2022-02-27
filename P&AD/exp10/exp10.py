import numpy as np

x1 = np. random. random( (100, 1))
y= 4 + 3*x1 + np. random. randn(100, 1)

x0 = np. ones ( (100, 1))
X = np. concatenate( (x0, x1), axis = 1)


#%%
temp1 = np.linalg.inv(np.dot (X.T, X))
temp2 = np.dot(temp1,X.T)
w = np. dot(temp2, y)
print("------------------------------")
print("Least squares method(Direct) Single Input")
print("------------------------------")
print("W0",w[0])
print("W1",w[1])


#%%
import numpy as np
x1 = np.random. random( (100, 3))
X=np.c_[np.ones((100,1)),x1]
a=[[4,5,8],
   [8,5,7],
   [7,6,3],
   [1,3,8]]
W=np.array(a)
y1=np.dot(X,W)
temp1 = np.linalg.inv(np.dot (X.T, X))
temp2 = np.dot(temp1,X.T)
w = np. dot(temp2, y1)
print("------------------------------")
print("Least squares method(Direct) Multiple Input")
print("------------------------------")
print("W1's are:\n" ,w)

#%%
X_ = np.random. random( (100, 3))
y1= 4 + 3*X_ + np. random. randn(100, 1)
y2= 5 + 2*X_ + np. random. randn(100, 1)
y3= 3 + 6*X_ + np. random. randn(100, 1)
y4= 7 + 9*X_ + np. random. randn(100, 1)
Xwb=np.c_[np.ones((100,1)),X_]
W_=Xwb.T.dot(Xwb)
tp1 = np.linalg.inv(np.dot (Xwb.T, Xwb))
tp2 = np.dot(tp1,Xwb.T)
W1 = np. dot(tp2, y1)
W2 = np. dot(tp2, y2)
W3 = np. dot(tp2, y3)
W4 = np. dot(tp2, y4)


print("------------Modified------------------")
print("Least squares method(Direct) Multiple Input")
print("------------------------------")
print("W1:\n",W1)
print("W2:\n",W2)
print("W3:\n",W3)
print("W4:\n",W4)
#%%
print(np.concatenate((W1,W2,W3,W4)))