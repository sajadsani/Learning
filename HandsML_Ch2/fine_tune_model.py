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
from sklearn.preprocessing import OrdinalEncoder, scale
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.model_selection import GridSearchCV
from scipy import stats

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
housing_num = housing.drop("ocean_proximity",axis=1) # drop makes a copy
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

# tune random forest model
forest_reg = RandomForestRegressor()
# define condidate paramters
param_grid = [{'n_estimators':[3,10,30],'max_features':[2,4,6,8]},
{'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]},]

grid_search = GridSearchCV(forest_reg,param_grid,cv=5,scoring='neg_mean_squared_error',
return_train_score=True)

grid_search.fit(housing_prepared,housing_labels)

# get best parameters
print(grid_search.best_params_)
print(grid_search.best_estimator_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"],cvres["params"]):
    print(np.sqrt(-mean_score),params)

# get the score of each attribute
features_importances = grid_search.best_estimator_.feature_importances_
extra_attribute = ["rooms_per_hhold","pop_per_hhold","bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs+extra_attribute+cat_one_hot_attribs
print(sorted(zip(features_importances,attributes),reverse=True))

# evaluate on test set
final_model=grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value",axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test,final_predictions)
print(np.sqrt(final_mse))

# get stat of generalization error

confidence = 0.95
squarred_error = (final_predictions - y_test)**2
np.sqrt(stats.t.interval(confidence,len(squarred_error)-1,loc=squarred_error.mean(),scale=
stats.sem(squarred_error)))

