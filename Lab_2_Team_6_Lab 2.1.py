import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, linear_model

# using the built in sklearn diabetes data set, create 3d index to show plots and linear regression line
diabetes = datasets.load_diabetes()
indices = (0, 2)

# train and test data
X_train = diabetes.data[:-8, indices]
X_test = diabetes.data[-8:, indices]
y_train = diabetes.target[:-8]
y_test = diabetes.target[-8:]

# using the linear model toolkit to create a preditciton plane in 3d
ols = linear_model.LinearRegression()
ols.fit(X_train, y_train)


# plot the data points, print test predict data in terminal
def plot_figures(fig_num, elev, azim, X_train, clf):
    fig = plt.figure(fig_num, figsize=(10, 10))
    plt.clf()
    ax = Axes3D(fig, elev=elev, azim=azim)
    ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c='r', marker='o')
    ax.plot_surface(np.array([[-.1, -.1], [.15, .15]]), np.array([[-.1, .15], [-.1, .15]]), clf.predict(np.array(
        [[-.1, -.1, .15, .15], [-.1, .15, -.1, .15]]).T).reshape((2, 2)), alpha=.9)

    ax.set_xlabel('X axis 1')
    ax.set_ylabel('X axis 2')
    ax.set_zlabel('Y')

    ax.w_xaxis.set_ticklabels([1,2,3,4,5,6,7,8])
    ax.w_yaxis.set_ticklabels([1,2,3,4,5,6,7,8])
    ax.w_zaxis.set_ticklabels([1,2,3,4,5,6,7,8])
    print(clf.predict(X_test))


# 3d axis points, pass the training to be plotted, pass 3d model angle,
elev = 18
azim = 133

plot_figures(1, elev, azim, X_train, ols)
plt.show()






