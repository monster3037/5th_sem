import pandas as pd
#%% Q!
fruits=pd.DataFrame({'Apples':[30],'Bananas':[21]})
print(fruits)

#%%  Q2
data={'Apples':[35,41],'Bananas':[21,34]}
fruit_sales=pd.DataFrame(data,index=['2017 Sales','2018 Sales'])
print(fruit_sales)

#%%  Q3
ingredients={'Dinner':pd.Series(['4 cups','1 cup','2 large','1 can']
                                ,index=['Flour','Milk','Eggs','Spam'])}
df=pd.DataFrame(ingredients)
print(df['Dinner'])

#%%  Q4
data=pd.read_csv('C:/Users/Dhruv Singhal/winemag-data_first150k.csv')
df=pd.DataFrame(data)
print(df)

#%%  Q5
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
print(animals)
print(type(animals))
animals.to_csv("B:/3rd year/5th sem/P&AD lab/hello.csv")

#%%
reviews=pd.read_csv('C:/Users/Dhruv Singhal/winemag-data_first150k.csv')
reviews.head()

#%%  Q!: Select the description column from reviews and assign the result to the variable desc
desc = reviews.description
print(desc)
#%%
#Q2: Select the first value from the description column of reviews , assigning it to the variable
first_desc=reviews.description.iloc[0]
print(first_desc)
#%%
#Q3: Select the first row of data from reviews , assigning it to the variable
first_row=reviews.iloc[0]
print(first_row)

#%%
#Q4: Select the first 10 values from desciption column in review, assigning the result to the variable
first_desc=reviews['description'].head(10)
print(first_desc)

#%%
#Q5: Select the records with index labels 1,2,3,5,8 assigning the result to variable
Sample_reviews=reviews.iloc[[1,2,3,5,8]]
print(Sample_reviews)

#%%
#Q6: Create a variable df containing the country,provience,region_1 and region_2 column of records
cols=['country','province','region_1','region_2']
indices=[0,1,10,100]
df=reviews.loc[indices,cols]
print(df)
#%%
#Q7: Create a variable containing the country and variety col of first 100 recors
cols_ine=['country','variety']
df=reviews.loc[:100,cols_ine]
print(df)
#%%
#Q8: create a dataframe italian_wines containing reviews of wine made3 in italy
italian_wine=reviews[reviews['country']=='Italy']
print(italian_wine)
#%%
#Q9:create a dataframe  containing all reviews with at least 95 points (out of 100) for wines from Australia or New Zealand.
top_oceania_wines = reviews.loc[reviews.country.isin(['Australia', 'New Zealand']) & (reviews.points >= 95)]
print(top_oceania_wines)

#%% region_1 and region_2 are pretty uninformative names for locale columns in the dataset. Create a copy of reviews with these columns renamed to region and locale, respectively.
renamed=reviews.rename(columns=dict(region_1='region',region_2='Locale'))
print(renamed.head())
#%%  Set the index name in the dataset to wines.
reindexed=reviews.rename_axis('wine',axis=0)
print(reindexed.head())

#%% What is the data type of the points column in the dataset?
print(reviews['points'].dtype)
#%%  Create a Series from entries in the points column, but convert the entries to strings. Hint: strings are str in native Python.
point_strings=reviews['points'].astype(str)
print(type(point_strings[0]))
#%% Sometimes the price column is null. How many reviews in the dataset are missing a price?
n_missing_prices=reviews['price'].isnull().sum()
print(n_missing_prices)
#%% What are the most common wine-producing regions? Create a Series counting the number of times each value occurs in the region_1 field. This field is often missing data, so replace missing values with Unknown. Sort in descending order. Your output should look something like this:
rev_pregion=reviews['region_1'].fillna('Unknown').value_counts().sort_values(ascending=False)
print(rev_pregion)

#%%What is the best wine I can buy for a given amount of money? Create a Series whose index is wine prices and whose values is the maximum number of points a wine costing that much was given in a review. Sort the values by price, ascending (so that 4.0 dollars is at the top and 3300.0 dollars is at the bottom).
best_rating_per_price = reviews.groupby('price')['points'].max().sort_index()
print(best_rating_per_price)

#%% What are the minimum and maximum prices for each variety of wine? Create a DataFrame whose index is the variety category from the dataset and whose values are the min and max values thereof.
price_extremes = reviews.groupby('variety').price.agg([min, max])
print(price_extremes)

#%%  What are the most expensive wine varieties? Create a variable sorted_varieties containing a copy of the dataframe from the previous question where varieties are sorted in descending order based on minimum price, then on maximum price (to break ties).
sorted_varieties = price_extremes.sort_values(by=['min', 'max'], ascending=False)   
print(sorted_varieties)

#%% What combination of countries and varieties are most common? Create a Series whose index is a MultiIndexof {country, variety} pairs. For example, a pinot noir produced in the US should map to {"US", "Pinot Noir"}. Sort the values in the Series in descending order based on wine count.
country_variety_counts = reviews.groupby(['country', 'variety']).size().sort_values(ascending=False)
print(country_variety_counts)







#%% What is the median of the points column in the reviews DataFrame?
med_points=reviews['points'].median()
print(med_points)

#%%What countries are represented in the dataset? (Your answer should not include any duplicates.)
countries=reviews.country.unique()
print(countries)

#%% How often does each country appear in the dataset? Create a Series reviews_per_country mapping countries to the count of reviews of wines from that country.

reviews_per_country=reviews.country.value_counts()
print(reviews_per_country)

#%% Create variable centered_price containing a version of the price column with the mean price subtracted.

centered_price = reviews.price- reviews.price.mean()
print(centered_price)

#%% I'm an economical wine buyer. Which wine is the "best bargain"? Create a variable bargain_wine with the title of the wine with the highest points-to-price ratio in the dataset.
bargain_idx=(reviews.points/reviews.price).idxmax()
bargain_wine = reviews.loc[bargain_idx]
print(bargain_wine)

#%% There are only so many words you can use when describing a bottle of wine. Is a wine more likely to be "tropical" or "fruity"? Create a Series descriptor_counts counting how many times each of these two words appears in the description column in the dataset.
n_trop = reviews['description'].map(lambda desc: "tropical" in desc ).sum()
n_fruity = reviews['description'].map(lambda desc: "fruity" in desc ).sum()
descriptor_counts=pd.Series([n_trop,n_fruity],index=['tropical','fruity'])
print(descriptor_counts)






























































