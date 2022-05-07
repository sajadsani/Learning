from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA, KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.manifold import LocallyLinearEmbedding


# Fetch data of mnist
mnist = fetch_openml('mnist_784',version=1)
X , y = mnist["data"],mnist["target"]
print(X.shape)
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
elif 1==0: # automatically uses d with 95% variance information
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X)
elif 1==0: # compress and decompress the data 
    pca = PCA(n_components = 154)
    X_reduced = pca.fit_transform(X)
    X_recovered = pca.inverse_transform(X_reduced)
elif 1==0: # incremental pca
    n_batches = 100
    if 1==1: # implementing inc_pca using for loop
        inc_pca = IncrementalPCA(n_components=154)
        for X_batch in np.array_split(X, n_batches):
            inc_pca.partial_fit(X_batch)
        X_reduced = inc_pca.transform(X)
    else: # implementing inc_pca using numpy memmap numpy data set , This part does not work
        X_mm = np.memmap(X, dtype="float32", mode="readonly", shape=(X.shape[0], X.shape[1]))
        batch_size = X.shape[0] // n_batches
        inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
        inc_pca.fit(X_mm)
elif 1==0: # kernel PCA, I get memory error
    rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.04)
    X_reduced = rbf_pca.fit_transform(X)
elif 1==0: # grid search for kpca which is looped with a classification
    clf = Pipeline([
    ("kpca", KernelPCA(n_components=2)),
    ("log_reg", LogisticRegression())
    ])
    param_grid = [{
    "kpca__gamma": np.linspace(0.03, 0.05, 10),
    "kpca__kernel": ["rbf", "sigmoid"]
    }]
    grid_search = GridSearchCV(clf, param_grid, cv=3)
    grid_search.fit(X, y)
    print(grid_search.best_params_)
elif 1==0: # inverse the data which is projected using KPCA, memory error
    rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433,
    fit_inverse_transform=True)
    X_reduced = rbf_pca.fit_transform(X)
    X_preimage = rbf_pca.inverse_transform(X_reduced)
    print(mean_squared_error(X, X_preimage))
elif 1==1: # implementing LLE, it takes a lot of time
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
    X_reduced = lle.fit_transform(X)



