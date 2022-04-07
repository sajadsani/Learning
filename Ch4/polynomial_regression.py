import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet




poly_features = PolynomialFeatures(degree=2, include_bias=False)

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
X_poly = poly_features.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print(lin_reg.intercept_, lin_reg.coef_)
# Learning curve To plot training and validation error for different trainins sizes
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
if 1==0: # linear regression
    lin_reg = LinearRegression()
    plot_learning_curves(lin_reg, X, y)
else: # polynomial regression
    polynomial_regression = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
    ])
    plot_learning_curves(polynomial_regression, X, y)
plt.show()

# Regularized regression
# ridge regression
if 1==0: # ridge regression
    ridge_reg = Ridge(alpha=1,solver="cholesky")
    ridge_reg.fit(X,y)
    print(ridge_reg.predict([[1.5]]))
elif 1==1: # stochastic gradient desent
    sgd_reg = SGDRegressor(penalty="l2")
    sgd_reg.fit(X, y.ravel())
    print(sgd_reg.predict([[1.5]]))
# lasso regression
if 1==0:
    lasso_reg = Lasso(alpha=0.1)
    lasso_reg.fit(X,y)
    print(lasso_reg.predict([[1.5]]))
else:
    sgd_reg_lasso = SGDRegressor(penalty="l1")
    sgd_reg_lasso.fit(X,y)
    print(sgd_reg_lasso.predict([[1.5]]))
# SelasticNet regression

Elasticnet_reg = ElasticNet(alpha=0.1,l1_ratio=0.5)
Elasticnet_reg.fit(X,y)
print(Elasticnet_reg.predict([[1.5]]))

