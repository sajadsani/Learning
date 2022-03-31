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

mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"] # X: (70000,784), y: (70000,1)

# turn y from string to integr
y = y.astype(np.uint8)

# splitting the set into test and training set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:] # X_train (60000,784) X_test (10000,784)

# binary training and test set
y_train_binary = (y_train == 5)
y_test_binary = (y_test == 5)

if 1==1: # train a binarry classifier
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train_binary)
    # to get the decision score for a sample:
    #y_score = sgd_clf.decision_function([X[0:1].to_numpy()])
    #print(y_score)

    # cross validation
    if 1 == 0:  # to do cross validation by hand
        skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        for train_index, test_index in skfolds.split(X_train, y_train_binary):
            # Construct a new unfitted estimator with the same parameters.
            clone_clf = clone(sgd_clf)
            X_train_folds = X_train.iloc[train_index].copy()
            y_train_folds = y_train_binary.iloc[train_index].copy()
            X_test_fold = X_train.iloc[test_index].copy()
            y_test_fold = y_train_binary.iloc[test_index].copy()
            clone_clf.fit(X_train_folds, y_train_folds)
            y_pred = clone_clf.predict(X_test_fold)
            n_correct = sum(y_pred == y_test_fold)
            print(n_correct / len(y_pred))
    else:  # to do cross validation using cross_val_score
        print(cross_val_score(sgd_clf, X_train,y_train_binary, cv=3, scoring="accuracy"))
else: # to define an estimator that predicts all the images as non-five
    class Never5Classifier(BaseEstimator):
        def fit(self, X, y=None):
            pass
        def predict(self, X):
            return np.zeros((len(X), 1), dtype=bool)
    never_5_clf = Never5Classifier()
    print(cross_val_score(never_5_clf, X_train, y_train_binary, cv=3, scoring="accuracy"))
# to obtain a predict on the training set which is clean, meaning that the prediction is made by a model
# that never saw the data during training
if 1==1: # confusion matrix of clean binary predictor on training set
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_binary, cv=3)
    # Confusion matrix:
    print(confusion_matrix(y_train_binary, y_train_pred))
    print(precision_score(y_train_binary, y_train_pred))
    print(recall_score(y_train_binary, y_train_pred))
    print(f1_score(y_train_binary, y_train_pred))
else: # confusion matrix of never_5_clssifier
    y_train_never5_pred = cross_val_predict(never_5_clf,X_train,y_train_binary,cv=3)
    print(confusion_matrix(y_train_binary, y_train_never5_pred))

# precision recal curve
y_scores = cross_val_predict(sgd_clf,X_train,y_train_binary,cv=3,method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_binary,y_scores)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plot_precision_recall_vs_threshold(precisions,recalls,thresholds)
plt.show()
# to set the threshold to 90% precision and predict based on that:
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)] # ~7816
print(threshold_90_precision)
# make predictions:
y_train_pred_90 = (y_scores >= threshold_90_precision)
print(precision_score(y_train_binary, y_train_pred_90))
print(recall_score(y_train_binary, y_train_pred_90))
# ROC curve, ROC AUC score
fpr, tpr, thresholds = roc_curve(y_train_binary,y_scores)
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
plot_roc_curve(fpr, tpr)
plt.show()
print(roc_auc_score(y_train_binary,y_scores))
# train and measure the ROC of randomforest classifier:
rf_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(rf_clf,X_train,y_train_binary,cv=3,method="predict_proba") #(60000,2) numpy
y_score_forest = y_probas_forest[:,1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_binary,y_score_forest)
plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show() 
