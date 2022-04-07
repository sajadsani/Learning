import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler




poly_features = PolynomialFeatures(degree=2, include_bias=False)

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
X_poly = poly_features.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print(lin_reg.intercept_, lin_reg.coef_)
# Learning curve To plot training and validation error for different trainins sizes
def plot_learning_curves(model, X, y):
    X_train_t, X_val_t, y_train_t, y_val_t = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train_t)):
        model.fit(X_train_t[:m], y_train_t[:m])
        y_train_predict = model.predict(X_train_t[:m])
        y_val_predict = model.predict(X_val_t)
        train_errors.append(mean_squared_error(y_train_t[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val_t, y_val_predict))
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

# early stopping code
poly_scaler = Pipeline([("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
("std_scaler", StandardScaler())])

X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)

sgd_reg_early_stoping = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,
penalty=None, learning_rate="constant", eta0=0.0005)

minimum_val_error = float("inf")
best_epoch = None
best_model = None

for epoch in range(1000):
    sgd_reg_early_stoping.fit(X_train_poly_scaled, y_train) # continues where it left off
    y_val_predict = sgd_reg_early_stoping.predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val, y_val_predict)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg_early_stoping)
