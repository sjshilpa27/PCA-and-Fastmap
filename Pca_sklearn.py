from sklearn.decomposition import PCA
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt #Used for plotting graph

# intialize pca and logistic regression model
pca = PCA(n_components=2)

# fit and transform data

#load data from pca_data text file
X = np.loadtxt("pca_data.txt")

#n_componentsint, float, None or str
#Number of components to keep. if n_components is not set all components are kept:
pca = PCA(n_components=2)
pca.fit(X)

#Transform the data from 3d to 2D
reduced_data = pca.transform(X)

# Function used for printing the 3D data
def plot3D(X):
    x_points = [each[0] for each in X]
    y_points = [each[1] for each in X]
    z_points = [each[2] for each in X]
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection="3d")
    ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv');
    plt.legend('3D Plot')

    plt.show()

#Function used for printing the 2D data
def plot2D(reduced_data):
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    for i in range(len(reduced_data)):
        _x = reduced_data[i][0]
        _y = reduced_data[i][1]

        plt.plot(_x, _y, 'b.', markersize=1)

    plt.show()

#Data  transformed to 2D
print("Transformed 2D array")
print(reduced_data)
#Some results to descrie the features of data
print("OUTPUT PARAMETERS")
print("Explained Variance")
print(pca.explained_variance_)
print("Singular values")
print(pca.singular_values_)
print("Get Covariance")
print(pca.get_covariance())
print("Get MEAN")
print(pca.mean_)
print("Eigen Vectors")
print(pca.components_)

plot3D(X)
plot2D(reduced_data)
