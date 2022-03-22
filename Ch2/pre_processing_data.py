from cProfile import label
from operator import index
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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
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

# seperate predictors and labels
housing = strat_train_set.drop("median_house_value",axis=1) # DROP makes a copy of data
housing_labels = strat_train_set["median_house_value"].copy()

# to fix missing data in "total_bedroom" attribute
if 1==0:
    # option 1: get ride of sample
    # # housing.dropna(subset=["total_bedrooms"])
    # option 2: get ride of attibute
    # # housing.drop("total_bedrooms",axis=1)
    # option 3: fill with median
    median = housing["total_bedrooms"].median()
    housing["total_bedrooms"].fillna(median,inplace=True)
else:
    # Fixing missing data using scikitlearn
    imputer = SimpleImputer(strategy="median")
    # to remove nun numeric attributes
    housing_num = housing.drop("ocean_proximity",axis=1) # drop makes a copy
    imputer.fit(housing_num)
    X = imputer.transform(housing_num) # it is a numpy data frame so convert it to pandas
    housing_tr = pd.DataFrame(X,columns=housing_num.columns, index=housing_num.index)

# handling text (categorical) attribute
housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(10))
ordinal_encoder = OrdinalEncoder()
housing_cat_encode = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encode[:10]) # it considers category 0 far from category 4 while category 4 is not further than category 1 from 0
# to encode with one binary encoder per attribute
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(np.shape(housing_cat_1hot))

# I did not understand this section, but it creates a class to combine attributes automatically
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# to define a pipeline of transformers to be implemented sequencially on data automatically
num_pipeline = Pipeline([('imputer',SimpleImputer(strategy="median")), # fill missing data
('attribute_adder',CombinedAttributesAdder()), # combine attributes
('std_scaler',StandardScaler()),]) # normalize the data (standardization)

housing_num_tr = num_pipeline.fit_transform(housing_num)

# to do the preprocessing on both numerical and categorical data at once
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([("num",num_pipeline,num_attribs),("cat",OneHotEncoder(),cat_attribs),])

housing_prepared = full_pipeline.fit_transform(housing)