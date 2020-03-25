'''

PA-3: Principal Component Analysis
Authors:
Amitabh Rajkumar Saini, amitabhr@usc.edu
Shilpa Jain, shilpaj@usc.edu
Sushumna Khandelwal, sushumna@usc.edu

Dependencies:
1. numpy : pip install numpy
2. matplotlib : pip install matplotlib
3. mplot3d : pip install mplot3d

Output:
Returns a 2D transformed data, writes model eigen values and transformed data on console and generates the plot of the same
(it takes time to generate 2D plot and 3d plot)

'''

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt #Used for plotting graph
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch


class Arrow3D(FancyArrowPatch):
    '''
    class to display eigen vectors on 3D plane
    '''
    def __init__(self, xs, ys, zs, *args, **kwargs):
        '''
        :param xs: mean of x,eigen value
        :param ys: mean of y,eigen value
        :param zs: mean of z,eigen value
        '''
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        '''

        :param renderer: to render points
        setting values of plot
        :return: returns nothing
        '''
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def plot_eigen_vectors_graph(all_samples, eig_vec_sc, mean_x, mean_y, mean_z):
    '''
    Displays a graph containing 3 eigen vectors
    :param all_samples: data matrix numpy array
    :param eig_vec_sc: eigen vectors numpy array
    :param mean_x: mean of column1
    :param mean_y: mean of column2
    :param mean_z: mean of column 3
    :return:  nothing
    '''
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(all_samples[0,:], all_samples[1,:], all_samples[2,:], 'o', markersize=8, color='green', alpha=0.2)
    ax.plot([mean_x], [mean_y], [mean_z], 'o', markersize=10, color='red', alpha=0.5)
    for v in eig_vec_sc.T:
        a = Arrow3D([mean_x, v[0]], [mean_y, v[1]], [mean_z, v[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
        ax.add_artist(a)
    ax.set_xlabel('x_values')
    ax.set_ylabel('y_values')
    ax.set_zlabel('z_values')

    plt.title('Eigenvectors')

    plt.show()

def plot3D(X):
    '''
    Displays a plot depicting 3d points on graph
    Return nothing
    '''
    x_points = [each[0] for each in X]
    y_points = [each[1] for each in X]
    z_points = [each[2] for each in X]
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection="3d")
    ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv');
    plt.legend('3D Plot')

    plt.show()

def plot2D(reduced_data):
    '''
    Displays a plot depicting 2d points on graph
    Return nothing
    '''

    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    for i in range(len(reduced_data)):
        _x = reduced_data[i][0]
        _y = reduced_data[i][1]

        plt.plot(_x, _y, 'y.', markersize=1)

    plt.show()


def pca(data_file,k):
    '''
    Runner Program
    :return: returns nothing
    '''

    #Data loaded in numpy array
    data=np.loadtxt(data_file,delimiter="\t")
    data=data-np.mean(data,axis=0)

    #numpy function to calculate covariance matrix
    short_cov = np.cov(data.T)

    #Eigen value and eigen vector calculation
    e_values , e_vectors = np.linalg.eig(short_cov)

    mean_val = np.mean(data, axis=0)
    mean_x, mean_y, mean_z = mean_val[0], mean_val[1], mean_val[2]

    plot_eigen_vectors_graph(data, e_vectors, mean_x, mean_y, mean_z)

    #Merging eigen value and eigen vector and sorting it into descending order.
    e_values_sorted = sorted(list(zip(e_values, e_vectors)), key=lambda x: x[0], reverse=True)
    print("Eigen Vectors")
    for i in e_vectors:
        print(i)
    #print(e_vectors)
    print("\n")
    print("Eigen Values")
    for i in e_values:
        print(i)

    #Picking k columns
    u_matrix = np.empty((data.shape[1], k))
    for i in range(len(e_values_sorted)):
        u_matrix[i] = e_values_sorted[i][1][:k]

    #Transformed 2d matrix
    z = np.dot(u_matrix.T, data.T)
    print("\n")
    print("Transformed Data into 2D")
    print(z.T)

    #Graph plot
    plot3D(data)
    plot2D(z.T)


#  print(e)
if __name__ == "__main__":
    pca('pca_data.txt',2)

