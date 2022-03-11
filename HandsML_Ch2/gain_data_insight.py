from cProfile import label
import os
import tarfile
from tkinter import S
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix

# direction to housing folder
Housing_path = r"C:\Users\aliak\OneDrive - HEC Montréal\Git_local\Learning\HandOn_CH2\dataset\housing"
# Function to load data from csv file:
def load_housing_data(path_to_folder):
    csv_path = os.path.join(path_to_folder,"housing.csv")
    return pd.read_csv(csv_path)
# to load data
housing = load_housing_data(Housing_path)

# to create test set with equal containing of all income categories
#make categories of each income group
housing["income_cat"] = pd.cut(housing["median_income"],bins=[0,1.5,3,4.5,6,np.inf],labels=[1,2,3,4,5])
#housing["income_cat"].hist()
# #plt.show()
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# to drop the category of income from data
for _set in (strat_train_set,strat_test_set):
    _set.drop("income_cat",axis=1,inplace=True)

# to get a copy from housing training set
housing = strat_train_set.copy()
#plot different figures
# housing.plot(kind="scatter",x="longitude",y="latitude")
# housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1)
housing.plot(kind="scatter", x="longitude",y="latitude",alpha=0.4, s=housing["population"]/100,
label="population",figsize=(10,7),c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True,)
plt.legend()
# # plt.show()

# to calculate the correlation matrix between each attribute and also pearson's r
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))
# to plot scatter matrix for important attributes
attributes = ["median_house_value","median_income","total_rooms","housing_median_age"]
scatter_matrix(housing[attributes],figsize=(12,8))
plt.show()

# have a closer look at the relationship between house value and income
housing.plot(kind="scatter",x="median_income",y="median_house_value",alpha=1)
plt.show()

# attribute combination for more meaningful attributes
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]
# look at again the correlation matrix
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))