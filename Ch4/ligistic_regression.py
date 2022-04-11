import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt



iris = datasets.load_iris()
X = iris["data"][:, 3:] # petal width, (150,1)

#print(np.unique(iris["target"]))
y = (iris["target"] == 2).astype(np.int) # 1 if Iris-Virginica, else 0, all values are 0,1,2

# train a ligistic regrression 01 class
log_reg = LogisticRegression()
log_reg.fit(X, y)
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
print(np.shape(X_new))
print(np.shape(y_proba))
print(y_proba[0:10])
plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")
plt.show()

# train a multi-class regression
X = iris["data"][:, (2, 3)] # petal length, petal width
y = iris["target"]
softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)
softmax_reg.fit(X, y)
print(softmax_reg.predict([[5, 2]]))
print(softmax_reg.predict_proba([[5, 2]]))