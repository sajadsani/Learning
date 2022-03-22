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
# direction to housing folder
Housing_path = r"C:\Users\aliak\OneDrive - HEC Montréal\Git_local\Learning\HandOn_CH2\dataset\housing"
# Function to load data from csv file:
def load_housing_data(path_to_folder):
    csv_path = os.path.join(path_to_folder,"housing.csv")
    return pd.read_csv(csv_path)
# to load data
housing = load_housing_data(Housing_path)
# Creating a function to split test_ration% of the data randomly
def split_train_test(data,test_ratio):
    # to fix the random seed
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    print(shuffled_indices)
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[: test_set_size]
    train_indices = shuffled_indices[test_set_size :]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing,0.2)
print(len(housing))
print(len(train_set))
print(len(test_set))

# to generate more solid test set and training set beased on data hash
def test_set_check(identifire,test_ratio):
    return crc32(np.int64(identifire)) & 0xffffffff < test_ratio*2**32

def split_train_test_byid(data,test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_,test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housind_with_id = housing.reset_index()
train_set_id, test_set_id = split_train_test_byid(housind_with_id,0.2,"index")

print(len(train_set_id))
print(len(test_set_id))

# Finally we can use sklearn to create test set and traint set
train_set_skl, test_set_skl = train_test_split(housing, test_size=0.2, random_state=42) 
print(len(train_set_skl))
print(len(test_set_skl))

# to create test set with equal containing of all income categories
#make categories of each income group
housing["income_cat"] = pd.cut(housing["median_income"],bins=[0,1.5,3,4.5,6,np.inf],labels=[1,2,3,4,5])
housing["income_cat"].hist()
#plt.show()
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print(strat_test_set["income_cat"].value_counts()/len(strat_test_set))

# to drop the category of income from data
for _set in (strat_train_set,strat_test_set):
    _set.drop("income_cat",axis=1,inplace=True)

    