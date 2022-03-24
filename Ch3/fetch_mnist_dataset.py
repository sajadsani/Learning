from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mnist = fetch_openml('mnist_784',version=1)
print(mnist.keys())
# having a look at date
X , y = mnist["data"],mnist["target"]
print(X.shape)
print(y.shape)
# to have a look at one image
some_digit = X[0:1]
some_digit = some_digit.to_numpy()


some_digit_image = some_digit.reshape(28,28)


plt.imshow(some_digit_image,cmap="binary")
plt.axis("off")
plt.show()
