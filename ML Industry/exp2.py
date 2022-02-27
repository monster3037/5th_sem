import pandas as pd
import numpy as np
#%%
fruits=pd.DataFrame({'Apples':[30,40],'Banana':[45,90]})
print(fruits)

#%%
fruits={'Apple':[80,90],'Banana':[45,50]}
df=pd.DataFrame(fruits,index=['Monaday 9 Aug','Tuesday 10 Aug'])
print(df)

#%%
data = {'Name':['A', 'B', 'C', 'D'],
        'Age':[27, 24, 22, 32],
        'Address':['Delhi', 'Kanpur', 'Allahabad', 'Kannauj'],
        'Qualification':['Msc', 'MA', 'MCA', 'Phd']}
df = pd.DataFrame(data)
print(df[['Name', 'Qualification']])
#%%
data=pd.read_csv("C:/Users/Dhruv Singhal/Social_Network_Ads.csv")
print(data.head)
#%%
data=pd.read_csv("C:/Users/Dhruv Singhal/Social_Network_Ads.csv",index_col="Gender")
first=data.loc['Male']
print(first)
#%%
data=pd.read_csv("C:/Users/Dhruv Singhal/Social_Network_Ads.csv",index_col="Gender")
first=data['Age']
print(first)

#%%
dict={"Breakfast":['Aloo Paratha',np.nan,'Milk','Cornflakes'],
      'Lunch': ['Dal','Rice',np.nan,'Salad'],
          "Dinner":['Panner',np.nan,'Naan','Salad']}
df=pd.DataFrame(dict)
print(df.isnull())

#%%
dict={"Breakfast":['Aloo Paratha',np.nan,'Milk','Cornflakes'],
      'Lunch': ['Dal','Rice',np.nan,'Salad'],
          "Dinner":['Panner',np.nan,'Naan','Salad']}
df=pd.DataFrame(dict)
print(df.fillna("not available at moment"))


#%%
dict={"Breakfast":['Aloo Paratha',np.nan,'Milk','Cornflakes'],
      'Lunch': ['Dal','Rice',np.nan,'Salad'],
          "Dinner":['Panner',np.nan,'Naan','Salad']}
df=pd.DataFrame(dict)
print(df.dropna())

#%%
