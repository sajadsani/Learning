from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
# Base class for all estimators in scikit-learn.
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"] # X: (70000,784), y: (70000,1)

# turn y from string to integr
y = y.astype(np.uint8)

# splitting the set into test and training set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:] # X_train (60000,784) X_test (10000,784)

# defining a SVM classifier
if 1==0: # defult value regarding OvO or OvR(OvA) strategies
    svm_clf = SVC()
    svm_clf.fit(X_train,y_train)
    print(svm_clf.predict(X[0:1].to_numpy()))
elif 1==0: # force svm classifier to use OvR strategy
    ovr_clf = OneVsRestClassifier(SVC())
    ovr_clf.fit(X_train,y_train)
    print(ovr_clf.predict(X[0:1].to_numpy()))
else: # train a stochastic gradian descent classifier
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train,y_train)
    print(sgd_clf.predict(X[0:1].to_numpy()))

# erro analysis: creating confusion matrix
y_train_pred = cross_val_predict(sgd_clf,X_train,y_train,cv=3)
conf_mx = confusion_matrix(y_train,y_train_pred)
print(conf_mx)
plt.matshow(conf_mx,cmap=plt.cm.gray)
plt.show()
# to normalize the  confusion matrix and emphasis on the errors
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()




