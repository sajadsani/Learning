import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures

X, y = make_moons(n_samples=100,noise=0.15)
#print(np.shape(X), np.shape(y)) # (100,2) (100,)
#print(X[0])
if 1==0: # Non-Linear SVM using adding polunomial features 
    polynomial_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=10, loss="hinge"))
    ])
    polynomial_svm_clf.fit(X, y)
    print(polynomial_svm_clf.predict([[10, 1]]))
elif 1==0: # polynomial kernel (kernel trick) 
    poly_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ])
    poly_kernel_svm_clf.fit(X, y)
    print(poly_kernel_svm_clf.predict([[10, 1]]))
elif 1==0: # Gaussian Radial Basis Finstion (RBF)
    rbf_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
    ])
    rbf_kernel_svm_clf.fit(X, y)
    print(rbf_kernel_svm_clf.predict([[10, 1]]))
elif 1==1: # Support Vector for Regression
    if 1==0: # Linear regression 
        svm_reg = LinearSVR(epsilon=1.5)
        svm_reg.fit(X, y)
        print(svm_reg.predict([[10, 1]]))
    else: # Non linear regression
        svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
        svm_poly_reg.fit(X, y)
        print(svm_poly_reg.predict([[10, 1]]))


