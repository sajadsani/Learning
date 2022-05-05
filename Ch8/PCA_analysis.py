from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.decomposition import PCA

# Fetch data of mnist
mnist = fetch_openml('mnist_784',version=1)
X , y = mnist["data"],mnist["target"]
if 1==0: # doing pca by hand
# normalization
    X_centered = X - X.mean(axis=0)
    # eighnvector calculauin
    U, s, Vt = np.linalg.svd(X_centered) # I get memory error
    # transform data to 2d dimention
    W2 = Vt.T[:, :2]
    X2D = X_centered.dot(W2)
elif 1==0: # doing pca using sikitlearn
    pca = PCA(n_components = 2)
    X2D = pca.fit_transform(X)
    print(pca.explained_variance_ratio_) # showing the ration of variance for each pc
elif 1==0: # calculating the number pf pcs containing 95% of the variance information
    pca = PCA()
    pca.fit(X)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95) + 1
    print(d)
elif 1==1: # automatically uses d with 95% variance information
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X)
