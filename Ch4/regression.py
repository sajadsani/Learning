import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor

num_samples = 100
X = 2*np.random.rand(num_samples,1)
y = 4 + 3*X + np.random.rand(num_samples,1)
# adding x0=1 to X
X_b = np.c_[np.ones((num_samples,1)),X]
if 1==0: # implementing LR using normal equation
    if 1==0: # implementing linear regression by hand
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        print(theta_best)
        # predict new sample
        X_new = np.array([[0], [2]])
        X_new_b = np.c_[np.ones((2, 1)), X_new] # add x0 = 1 to each instance
        y_predict = X_new_b.dot(theta_best)
        print(y_predict)
        # plot the predictions and samples
        plt.plot(X_new, y_predict, "r-")
        plt.plot(X, y, "b.")
        plt.axis([0, 2, 0, 15])
        plt.show()
    else: # implementing linear regression by scikitlearn
        lin_reg = LinearRegression()
        lin_reg.fit(X, y)
        print(lin_reg.intercept_, lin_reg.coef_)
        X_new = np.array([[0], [2]])
        print(lin_reg.predict(X_new))
elif 1==0: # implementing LR using gradient descent
    eta = 0.1 # learning rate
    n_iterations = 1000
    theta = np.random.randn(2,1) # random initialization
    for iteration in range(n_iterations):
        gradients = 2/num_samples * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
    print(theta)
elif 1==1: # implementing LR using Stochastic gradient descent by hand 
    if 1==0: # implementing SGD by hand
        n_epochs = 50
        t0, t1 = 5, 50 # learning schedule hyperparameters  
        def learning_schedule(t):
            return t0 / (t + t1)
        theta = np.random.randn(2,1) # random initialization
        for epoch in range(n_epochs):
            for i in range(num_samples):
                random_index = np.random.randint(num_samples)
                xi = X_b[random_index:random_index+1]
                yi = y[random_index:random_index+1]
                gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
                eta = learning_schedule(epoch * num_samples + i)
                theta = theta - eta * gradients 
        print(theta) 
    else:
        sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
        sgd_reg.fit(X, y.ravel())    
        print(sgd_reg.intercept_, sgd_reg.coef_)
