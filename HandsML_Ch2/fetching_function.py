
import os
import tarfile
import urllib.request

Download_root = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
Housing_path = r"C:\Users\aliak\OneDrive - HEC Montréal\Git_local\Learning\HandOn_CH2\dataset\housing"
housing_url = Download_root + "datasets/housing/housing.tgz"

def fetch_housing_data(h_url,h_path):
    os.makedirs(h_path,exist_ok=True)
    tgz_path = os.path.join(h_path,"housing.tgz")
    urllib.request.urlretrieve(h_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=h_path)
    housing_tgz.close()

fetch_housing_data(housing_url,Housing_path)