import numpy as np
#%% Insert the correct method for creating a NumPy array.
arr = np.array([1, 2, 3, 4, 5])
print(arr)
#%% Insert the correct argument for creating a NumPy array with 2 dimensions.
arr = np.array([1, 2, 3, 4], ndmin=2)
print(arr)
#%% Insert the correct syntax for checking the number of dimension of a NumPy array.
arr = np.array([1, 2, 3, 4])
print(arr.ndim)
#%% Insert the correct syntax for printing the first item in the array.
arr = np.array([1, 2, 3, 4, 5])
print(arr[0])
#%% Insert the correct syntax for printing the number 50 from the array.
arr = np.array([1, 2, 3, 4, 5])
print(arr[4])
#%% Insert the correct syntax for printing the number 50 from the array.
arr = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
print(arr[1,0])
#%% Use negative index to print the last item in the array.
arr = np.array([10, 20, 30, 40, 50])
print(arr[-1])
#%% Insert the correct slicing syntax to print the following selection of the array:
#Everything from (including) the second item to (not including) the fifth item.
arr = np.array([10, 15, 20, 25, 30, 35, 40])
print(arr[1:4])
#%% Insert the correct slicing syntax to print the following selection of the array:
#Everything from (including) the third item to (not including) the fifth item.
arr = np.array([10, 15, 20, 25, 30, 35, 40])
print(arr[2:4])
#%% Insert the correct slicing syntax to print the following selection of the array:
#Every other item from (including) the second item to (not including) the fifth item.
arr = np.array([10, 15, 20, 25, 30, 35, 40])
print(arr[1:5:2])
#%% Insert the correct slicing syntax to print the following selection of the array:
#Every other item from the entire array
arr = np.array([10, 15, 20, 25, 30, 35, 40])
print(arr[::2])
#%% NumPy uses a character to represent each of the following data types, which one?
# i = integer
# b = boolean
# u = unsigned integer
# f= float
# c = complex float
# m = timedelta
# M = datatime
# O= object
# S = string
#%% Insert the correct NumPy syntax to print the data type of an array.
arr = np.array([1, 2, 3, 4])
print(arr.dtype)
#%% Insert the correct argument to specify that the array should be of type STRING.
arr = np.array([1, 2, 3, 4], dtype='S')
print(arr)
#%% Insert the correct method to change the data type to integer.
arr = np.array([1.1, 2.1, 3.1])
newarr = arr.astype('i')
print(newarr)
#%% Use the correct method to make a copy of the array.
arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
print(x)
#%% Use the correct method to make a view of the array.
arr = np.array([1, 2, 3, 4, 5])
x = arr.view()
print(x)
#%% Use the correct NumPy syntax to check the shape of an array.
arr = np.array([1, 2, 3, 4, 5])
print(arr.shape)
#%% Use the correct NumPy method to change the shape of an array from 1-D to 2-D.
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
newarr = arr.reshape(4, 3)
print(newarr)
#%% Use a correct NumPy method to change the shape of an array from 2-D to 1-D.
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
newarr = arr.reshape(-1)
print(newarr)
#%% Use a correct NumPy method to join two arrays into a single array.
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.concatenate((arr1, arr2))
print(arr)
#%%  Use the correct NumPy method to find all items with the value 4.
arr = np.array([1, 2, 3, 4, 5, 4, 4])
x = np.where(arr == 4)
print(x)
#%% Use the correct NumPy method to return a sorted array.
arr = np.array([3, 2, 0, 1])
x = np.sort(arr)
print(x)






