import os
import tarfile
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
# direction to housing folder
Housing_path = r"C:\Users\aliak\OneDrive - HEC Montréal\Git_local\Learning\HandOn_CH2\dataset\housing"
# Function to load data from csv file:
def load_housing_data(path_to_folder):
    csv_path = os.path.join(path_to_folder,"housing.csv")
    return pd.read_csv(csv_path)
# to load data
housing = load_housing_data(Housing_path)
# to watch the headlines 
print(housing.head())
# to get idea about the data
housing.info()
# to get better idea about the data in column "ocean_proximity"
print(housing["ocean_proximity"].value_counts())
# method describe
print(housing.describe())
# to draw diagram over data frame housing
housing.hist(bins=50,figsize=(20,15))
plt.show()

