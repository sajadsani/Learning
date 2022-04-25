from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


X, y = make_moons(n_samples=120,noise=0.4)
X_train, X_test, y_train, y_test = X[:100], X[100:], y[:100], y[100:]
if 1==0: # train different classifiers
    log_clf = LogisticRegression()
    rnd_clf = RandomForestClassifier()
    svm_clf = SVC()

    voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard')

    voting_clf.fit(X_train, y_train)
    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

elif 1==0:# Bagging and pasting with decision tree classifier
    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=500,
        max_samples=100, bootstrap=True, n_jobs=-1, oob_score=True)
    bag_clf.fit(X_train, y_train)
    print(bag_clf.oob_score_)
    y_pred = bag_clf.predict(X_test)
    print( accuracy_score(y_test, y_pred))
    print(bag_clf.oob_decision_function_)
elif 1==1: 
    rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
    rnd_clf.fit(X_train, y_train)
    y_pred_rf = rnd_clf.predict(X_test)
    print( accuracy_score(y_test, y_pred_rf))



