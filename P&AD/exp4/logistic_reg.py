import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import seaborn as sns
a=make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_classes=2,n_clusters_per_class=1)
f1=[]
f2=[]
label=[]
for i in range(len(a[0])):
    f1.append(a[0][i][0])
    f2.append(a[0][i][1])
    label.append(a[1][i])
x={'f1':f1,'F2':f2}
dataset=pd.DataFrame(x)
y={'label':label}
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.3,random_state=42)