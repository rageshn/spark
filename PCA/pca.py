import os
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from numpy.linalg import eigh
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import get_cmap
import matplotlib.cm as cm


# This function displays the data using matplotlib
def prepare_plot(xticks, yticks, figsize=(10.5, 6), hide_labels=False, grid_color='#999999',
                 grid_width=1.0):
    """Template for generating the plot layout."""
    plt.close()
    fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
    ax.axes.tick_params(labelcolor='#999999', labelsize='10')
    for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
        axis.set_ticks_position('none')
        axis.set_ticks(ticks)
        axis.label.set_color('#999999')
        if hide_labels:
            axis.set_ticklabels([])
    plt.grid(color=grid_color, linewidth=grid_width, linestyle='-')
    map(lambda position: ax.spines[position].set_visible(False), ['bottom', 'top', 'left', 'right'])
    return fig, ax


# Create 2D Gaussian Distribution with random data.
def create_2d_gaussian(mn, variance, cov, n):
    """Randomly sample points from a two-dimensional Gaussian distribution"""
    np.random.seed(142)
    return np.random.multivariate_normal(np.array([mn, mn]), np.array([[variance, cov], [cov, variance]]), n)


# Create a dataset with 2D gaussian distribution with mean = 50 & covariance = 0 - This creates a spherical
data_random = create_2d_gaussian(mn=50, variance=1, cov=0, n=100)

# generate layout and plot data
# fig, ax = prepare_plot(np.arange(46, 55, 2), np.arange(46, 55, 2))
# ax.set_xlabel(r'Simulated $x_1$ values'), ax.set_ylabel(r'Simulated $x_2$ values')
# ax.set_xlim(45, 54.5), ax.set_ylim(45, 54.5)
# plt.scatter(data_random[:,0], data_random[:,1], s=14**2, c='#d6ebf2', edgecolors='#8cbfd0', alpha=0.75)
# plt.show()

# Create a dataset with 2D gaussian distribution with mean = 50 & covariance = 0.9
data_correlated = create_2d_gaussian(mn=50, variance=1, cov=.9, n=100)

# generate layout and plot data
# fig, ax = prepare_plot(np.arange(46, 55, 2), np.arange(46, 55, 2))
# ax.set_xlabel(r'Simulated $x_1$ values'), ax.set_ylabel(r'Simulated $x_2$ values')
# ax.set_xlim(45.5, 54.5), ax.set_ylim(45.5, 54.5)
# plt.scatter(data_correlated[:,0], data_correlated[:,1], s=14**2, c='#d6ebf2', edgecolors='#8cbfd0', alpha=0.75)
# plt.show()


# With mean = 50 and cov = 0, data is uncorrelated and it looks spherical.
# But with cov = 0.9, the data is positively correlated.
# The non-spherical data is amenable to dimensionality reduction via PCA, while the spherical data is not.

# Create a SparkSession ad Spark Context
# Create an RDD with correlated data.
spark = SparkSession.builder.appName("PCA").master("local").getOrCreate()
sc = spark.sparkContext
correlated_data = sc.parallelize(data_correlated)


""" 
Interpreting PCA
---------------------

PCA can be interpreted as identifying the "directions" along which the data vary the most.
In the first step of PCA, first center our data. Working with our correlated dataset,
    first compute the mean of each feature (column) in the dataset.
Then for each observation, modify the features by subtracting their corresponding mean, to create a zero mean dataset.

"""

# mean_correlated = correlated_data.mean()
# correlated_data_zero_mean = correlated_data.map(lambda x: x - mean_correlated)

"""
Sample Covariance matrix
------------------------

If X => R^(nXd) is defined as the zero mean data matrix, then the sample covariance matrix is defined as Cx = (1/n)((X^T).X)

Compute the outer product of each data point, add together these outer products and divide by number of data points.

"""

# correlated_cov = correlated_data_zero_mean.map(lambda x: np.outer(x.T, x)).reduce(lambda x, y: x+y)
#                       / correlated_data_zero_mean.count()


"""
Covariance function
--------------------
"""

def estimate_covariance(data):
    """Compute the covariance matrix for a given rdd.

    Note:
        The multi-dimensional covariance array should be calculated using outer products.  Don't
        forget to normalize the data by first subtracting the mean.

    Args:
        data (RDD of np.ndarray):  An `RDD` consisting of NumPy arrays.

    Returns:
        np.ndarray: A multi-dimensional array where the number of rows and columns both equal the
            length of the arrays in the input `RDD`.
    """
    mean_data = data.mean()
    data_zero_mean = data.map(lambda x: x - mean_data)
    multi_dim_covariance = data_zero_mean.map(lambda x: np.outer(x.T, x)).reduce(lambda x, y: x+y)/data_zero_mean.count()
    return multi_dim_covariance


# correlated_cov_auto = estimate_covariance(correlated_data)

"""
Eigen Decomposition
--------------------
From the co-variance matrix, we can find the directions of maximal variance in the data. 
Eigenvalues and eigenvectores can be calculated by performing an eigendecomposition of the matrix. 
The 'd' eigenvectors of the cov matrix gives the direction of maximal variance, i.e., principal components. 
The associated eigenvalues are the variances in these directions. 
In particular, the eigen vector corresponding to the largest eigenvalue is the direction of maximal variance (this is sometimes called the "top" eigenvector). 
Eigendecomposition of a d×d covariance matrix has a (roughly) cubic runtime complexity with respect to 'd'. 
Whenever 'd' is relatively small (e.g., less than a few thousand) eigendecomposition can be performed locally.

Steps:
    1. Perform eigendecomposition
    2. Sort the eigen vectors based on their corresponding eigen values (from high to low), yielding a matrix where columns are the eigen vectors,
        and the first column is the top eigen vector. 
    3. Obtain the indices of the eigen values that correspond to the ascending value of the eigen value. 
"""

# eig_vals, eig_vecs = eigh(correlated_cov_auto)
# print('eigenvalues: {0}'.format(eig_vals))
# print('\neigenvectors: \n{0}'.format(eig_vecs))

# inds = np.argsort(eig_vals)
# print('\nIndices: ', inds)
# top_value = np.flipud(inds)[0]
# print('\nTop value: ', top_value)
# top_component = eig_vecs[top_value]
# print('\nTop principal component: {0}'.format(top_component))


"""
PCA Scores
-----------
We have to use the pca to compute the 1D representation of the original data. 
Calculate the dot product between each data point in the raw data and the top principal component. 
"""

# correlated_data_scores = correlated_data.map(lambda x: np.dot(x, top_component))

"""
PCA Function
-------------

This function computes the top 'k' principal components and principal scores for a given dataset. 
The top 'k' PC will be returned in descending order when ranked by their corresponding principal scores. 
"""

def pca(data, k=2):
    """Computes the top `k` principal components, corresponding scores, and all eigenvalues.

    Note:
        All eigenvalues should be returned in sorted order (largest to smallest). `eigh` returns
        each eigenvectors as a column.  This function should also return eigenvectors as columns.

    Args:
        data (RDD of np.ndarray): An `RDD` consisting of NumPy arrays.
        k (int): The number of principal components to return.

    Returns:
        tuple of (np.ndarray, RDD of np.ndarray, np.ndarray): A tuple of (eigenvectors, `RDD` of
            scores, eigenvalues).  Eigenvectors is a multi-dimensional array where the number of
            rows equals the length of the arrays in the input `RDD` and the number of columns equals
            `k`.  The `RDD` of scores has the same number of rows as `data` and consists of arrays
            of length `k`.  Eigenvalues is an array of length d (the number of features).
    """
    multi_dim_cov = estimate_covariance(data)
    eig_values, eig_vectors = eigh(multi_dim_cov)
    # Return the `k` principal components, `k` scores, and all eigenvalues
    inds = np.argsort(eig_values)
    reversed_inds = np.flipud(inds)
    top_indices = reversed_inds[:k]
    top_component = eig_vectors[:,top_indices]
    correlated_data_scores = data.map(lambda x : np.dot(x, top_component))
    eig_value_reversed = eig_values[::-1]
    return (top_component, correlated_data_scores, eig_value_reversed)


# Run pca on correlated_data with k = 2
# top_components_correlated, correlated_data_scores_auto, eigenvalues_correlated = pca(correlated_data)

# print("Top component: {0}".format(top_components_correlated))
# print("Scores: {0}".format(correlated_data_scores_auto))
# print("Eig Correlated: {0}".format(eigenvalues_correlated))
# Note that the 1st principal component is in the first column
# print('top_components_correlated: \n{0}'.format(top_components_correlated))
# print ('\ncorrelated_data_scores_auto (first three): \n{0}'.format('\n'.join(map(str, correlated_data_scores_auto.take(3)))))
# print('\neigenvalues_correlated: \n{0}'.format(eigenvalues_correlated))

# Create a higher dimensional test set
# pca_test_data = sc.parallelize([np.arange(x, x + 4) for x in np.arange(0, 20, 4)])
# components_test, test_scores, eigenvalues_test = pca(pca_test_data, 3)

# print('\npca_test_data: \n{0}'.format(np.array(pca_test_data.collect())))
# print('\ncomponents_test: \n{0}'.format(components_test))
# print('\ntest_scores (first three): \n{0}'.format('\n'.join(map(str, test_scores.take(3)))))
# print('\neigenvalues_test: \n{0}'.format(eigenvalues_test))


"""
PCA on Random Data
------------------
"""

# random_data_rdd = sc.parallelize(data_random)

# Use pca on data_random
# top_components_random, random_data_scores_auto, eigenvalues_random = pca(random_data_rdd)

# print('top_components_random: \n{0}'.format(top_components_random))
# print('\nrandom_data_scores_auto (first three): \n{0}'.format('\n'.join(map(str, random_data_scores_auto.take(3)))))
# print('\neigenvalues_random: \n{0}'.format(eigenvalues_random))


"""
PCA Projection
--------------
Plot the original data and the 1-dimensional reconstruction using the top principal component to see how the PCA solution looks. 
The original data is plotted as before; however, the 1-dimensional reconstruction (projection) is plotted in green 
    on top of the original data and the vectors (lines) representing the two principal components are shown as dotted lines.
"""

def project_points_and_get_lines(data, components, x_range):
    """Project original data onto first component and get line details for top two components."""
    top_component = components[:, 0]
    slope1, slope2 = components[1, :2] / components[0, :2]

    means = data.mean()[:2]
    demeaned = data.map(lambda v: v - means)
    projected = demeaned.map(lambda v: (v.dot(top_component) /
                                        top_component.dot(top_component)) * top_component)
    remeaned = projected.map(lambda v: v + means)
    x1,x2 = zip(*remeaned.collect())

    line_start_P1_X1, line_start_P1_X2 = means - np.asarray([x_range, x_range * slope1])
    line_end_P1_X1, line_end_P1_X2 = means + np.asarray([x_range, x_range * slope1])
    line_start_P2_X1, line_start_P2_X2 = means - np.asarray([x_range, x_range * slope2])
    line_end_P2_X1, line_end_P2_X2 = means + np.asarray([x_range, x_range * slope2])

    return ((x1, x2), ([line_start_P1_X1, line_end_P1_X1], [line_start_P1_X2, line_end_P1_X2]),
            ([line_start_P2_X1, line_end_P2_X1], [line_start_P2_X2, line_end_P2_X2]))


# ((x1, x2), (line1X1, line1X2), (line2X1, line2X2)) = project_points_and_get_lines(correlated_data, top_components_correlated, 5)

# generate layout and plot data
# fig, ax = prepare_plot(np.arange(46, 55, 2), np.arange(46, 55, 2), figsize=(7, 7))
# ax.set_xlabel(r'Simulated $x_1$ values'), ax.set_ylabel(r'Simulated $x_2$ values')
# ax.set_xlim(45.5, 54.5), ax.set_ylim(45.5, 54.5)
# plt.plot(line1X1, line1X2, linewidth=3.0, c='#8cbfd0', linestyle='--')
# plt.plot(line2X1, line2X2, linewidth=3.0, c='#d6ebf2', linestyle='--')
# plt.scatter(data_correlated[:,0], data_correlated[:,1], s=14**2, c='#d6ebf2',edgecolors='#8cbfd0', alpha=0.75)
# plt.scatter(x1, x2, s=14**2, c='#62c162', alpha=.75)
# plt.show()


# ((x1, x2), (line1X1, line1X2), (line2X1, line2X2)) = project_points_and_get_lines(random_data_rdd, top_components_random, 5)

# generate layout and plot data
# fig, ax = prepare_plot(np.arange(46, 55, 2), np.arange(46, 55, 2), figsize=(7, 7))
# ax.set_xlabel(r'Simulated $x_1$ values'), ax.set_ylabel(r'Simulated $x_2$ values')
# ax.set_xlim(45.5, 54.5), ax.set_ylim(45.5, 54.5)
# plt.plot(line1X1, line1X2, linewidth=3.0, c='#8cbfd0', linestyle='--')
# plt.plot(line2X1, line2X2, linewidth=3.0, c='#d6ebf2', linestyle='--')
# plt.scatter(data_random[:,0], data_random[:,1], s=14**2, c='#d6ebf2', edgecolors='#8cbfd0', alpha=0.75)
# plt.scatter(x1, x2, s=14**2, c='#62c162', alpha=.75)
# plt.show()


"""
3D Data Visualization
----------------------
"""

"""
m = 100
mu = np.array([50, 50, 50])
r1_2 = 0.9
r1_3 = 0.7
r2_3 = 0.1
sigma1 = 5
sigma2 = 20
sigma3 = 20
c = np.array([[sigma1 ** 2, r1_2 * sigma1 * sigma2, r1_3 * sigma1 * sigma3],
             [r1_2 * sigma1 * sigma2, sigma2 ** 2, r2_3 * sigma2 * sigma3],
             [r1_3 * sigma1 * sigma3, r2_3 * sigma2 * sigma3, sigma3 ** 2]])
np.random.seed(142)
data_threeD = np.random.multivariate_normal(mu, c, m)


norm = Normalize()
cmap = get_cmap("Blues")
clrs = cmap(np.array(norm(data_threeD[:,2])))[:,0:3]

fig = plt.figure(figsize=(11, 6))
ax = fig.add_subplot(121, projection='3d')
ax.azim=-100
ax.scatter(data_threeD[:,0], data_threeD[:,1], data_threeD[:,2], c=clrs, s=14**2)

xx, yy = np.meshgrid(np.arange(-15, 10, 1), np.arange(-50, 30, 1))
normal = np.array([0.96981815, -0.188338, -0.15485978])
z = (-normal[0] * xx - normal[1] * yy) * 1. / normal[2]
xx = xx + 50
yy = yy + 50
z = z + 50

ax.set_zlim((-20, 120)), ax.set_ylim((-20, 100)), ax.set_xlim((30, 75))
ax.plot_surface(xx, yy, z, alpha=.10)

ax = fig.add_subplot(122, projection='3d')
ax.azim=10
ax.elev=20
#ax.dist=8
ax.scatter(data_threeD[:,0], data_threeD[:,1], data_threeD[:,2], c=clrs, s=14**2)

ax.set_zlim((-20, 120)), ax.set_ylim((-20, 100)), ax.set_xlim((30, 75))
ax.plot_surface(xx, yy, z, alpha=.1)
plt.tight_layout()
plt.show()
"""

"""
3D to 2D
--------
Use PCA to see if we can recover the 2-dimensional plane on which the data live. 
Parallelize the data, and use our PCA function from above, with k=2 components.
"""

# threeD_data = sc.parallelize(data_threeD)
# components_threeD, threeD_scores, eigenvalues_threeD = pca(threeD_data)

# print('components_threeD: \n{0}'.format(components_threeD))
# print('\nthreeD_scores (first three): \n{0}'.format('\n'.join(map(str, threeD_scores.take(3)))))
# print('\neigenvalues_threeD: \n{0}'.format(eigenvalues_threeD))


"""
Variance explained
-------------------
Let's quantify how much of variance is captured by PCA in each of 3 synthetic datasets. 
Compute the fraction of retained variance by the top principal components. 
Recall that the eigenvalue corresponding to each principal component captures the variance along this direction.
If our initial data is d-dimensional, then the total variance in our data equals: ∑ i=1 -> d  λ(i) 
    where λ(i) is the eigenvalue corresponding to the i'th principal component. 
Moreover, if we use PCA with some k<d, then we can compute the variance retained by these principal components by adding the top k eigenvalues. 
The fraction of retained variance equals the sum of the top k eigenvalues divided by the sum of all of the eigenvalues.
"""


def variance_explained(data, k=1):
    """Calculate the fraction of variance explained by the top `k` eigenvectors.

    Args:
        data (RDD of np.ndarray): An RDD that contains NumPy arrays which store the
            features for an observation.
        k: The number of principal components to consider.

    Returns:
        float: A number between 0 and 1 representing the percentage of variance explained
            by the top `k` eigenvectors.
    """
    components, scores, eigenvalues = pca(data, k)
    top_k_eigval = np.sum(eigenvalues[:k])
    total_sum_eigval = np.sum(eigenvalues)
    return top_k_eigval / total_sum_eigval


# variance_random_1 = variance_explained(random_data_rdd, 1)
# variance_correlated_1 = variance_explained(correlated_data, 1)
# variance_random_2 = variance_explained(random_data_rdd, 2)
# variance_correlated_2 = variance_explained(correlated_data, 2)
# variance_threeD_2 = variance_explained(threeD_data, 2)
# print('\nPercentage of variance explained by the first component of random_data_rdd: {0:.1f}%'.format(variance_random_1 * 100))
# print('Percentage of variance explained by both components of random_data_rdd: {0:.1f}%'.format(variance_random_2 * 100))
# print('\nPercentage of variance explained by the first component of correlated_data: {0:.1f}%'.format(variance_correlated_1 * 100))
# print('Percentage of variance explained by both components of correlated_data: {0:.1f}%'.format(variance_correlated_2 * 100))
# print('\nPercentage of variance explained by the first two components of threeD_data: {0:.1f}%'.format(variance_threeD_2 * 100))


"""
Parse, Inspect and Pre-process Neuroscience data
-------------------------------------------------

Light sheet microscopy is a technique of recording neural activity as images. This is done in a transparent animal, larval zebra fish over nearly its entire brain.
The resulting data is a time-varying images containing activity of hundreds of thousands of neurons. 
Given the raw data, find compact spatial and temporal patterns: 
    1. Which group of neurons are active together?
    2. What is the time course of their activity?
    3. Are those patterns specific to particular events happening during the experiments? 

PCA is a powerful technique for finding spatial and temporal patterns in these data. 
"""

"""
Load Neuroscience data
-----------------------

The data is stored as a text file. Each line contains the time series of image intensity for a single pixel in a time varying image. 
The first two number in each line are the spatial co-ordinates of the pixel and the remaining numbers are the time series. 
"""
input_path = "/home/ragesh/Data/PCA/neuro.txt"
lines = sc.textFile(input_path)
# print('\nNeuro Data sample: ', lines.first()[0:100])


"""
Parsing the data
-----------------

Parse the data into key value representation.
Key: Tuple of 2D spacial co-ordinates
Value: Numpy array storing associated time-series. 
"""


def parse_neuro_data(line):
    """Parse the raw data into a (`tuple`, `np.ndarray`) pair.

    Note:
        You should store the pixel coordinates as a tuple of two ints and the elements of the pixel intensity
        time series as an np.ndarray of floats.

    Args:
        line (str): A string representing an observation.  Elements are separated by spaces.  The
            first two elements represent the coordinates of the pixel, and the rest of the elements
            represent the pixel intensity over time.

    Returns:
        tuple of tuple, np.ndarray: A (coordinate, pixel intensity array) `tuple` where coordinate is
            a `tuple` containing two values and the pixel intensity is stored in an NumPy array
            which contains 240 values.
    """
    words = line.split(" ")
    co_ord = tuple(int(x) for x in words[0:2])
    pixel_intensity = np.array(words[2:], dtype=float)
    return (co_ord, pixel_intensity)


# Parse each line of data
raw_data = lines.map(parse_neuro_data)
raw_data.cache()
entry = raw_data.first()
print('\nLength of movie is {0} seconds'.format(len(entry[1])))
print('\nNumber of pixels in movie is {0:,}'.format(raw_data.count()))
print('\nFirst entry of raw_data (with only the first five values of the NumPy array):\n({0}, {1})'.format(entry[0], entry[1][:5]))


"""
Max and min fluorescence
-------------------------

The raw time series data are in the units of image fluorescence, and baseline fluorescence varies somewhat arbitrarily from pixel to pixel.
"""

# Get the minimum and maximum
# mn = raw_data.map(lambda x: np.ndarray.min(x[1])).reduce(lambda x, y: x if x < y else y)
# mx = raw_data.map(lambda x: np.ndarray.max(x[1])).reduce(lambda x, y: y if x < y else x)
# print(mn, mx)

# visualize a pixel that exhibits a standard deviation of over 100.

# def get_std(line):
#     if np.std(line[1]) > 100:
#         return line


# example = raw_data.filter(get_std).values().first()

# generate layout and plot data
# fig, ax = prepare_plot(np.arange(0, 300, 50), np.arange(300, 800, 100))
# ax.set_xlabel(r'time'), ax.set_ylabel(r'fluorescence')
# ax.set_xlim(-20, 270), ax.set_ylim(270, 730)
# plt.plot(range(len(example)), example, c='#8cbfd0', linewidth='3.0')
# plt.show()


"""
Fractional signal change
-------------------------

Convert the raw fluorescence units to more intuitive fractional signal change.
For all the pixels, time series for a particular pixel should be subtracted and divided by the mean.
"""

def rescale(ts):
    """Take a np.ndarray and return the standardized array by subtracting and dividing by the mean.

    Note:
        You should first subtract the mean and then divide by the mean.

    Args:
        ts (np.ndarray): Time series data (`np.float`) representing pixel intensity.

    Returns:
        np.ndarray: The times series adjusted by subtracting the mean and dividing by the mean.
    """
    mean = np.ndarray.mean(ts)
    # print(mean)
    return (ts - mean)/mean


# Scale the values for each pixel entry
scaled_data = raw_data.mapValues(lambda x: rescale(x))
# print(scaled_data.first())
mn_scaled = scaled_data.map(lambda x: np.ndarray.min(x[1])).min()
mx_scaled = scaled_data.map(lambda x: np.ndarray.max(x[1])).max()
# print(mn_scaled, mx_scaled)


"""
Normalized data
---------------
"""
# visualize a pixel that exhibits a standard deviation of over 0.1.

# def get_std_scaled(line):
#      if np.std(line[1]) > 0.1:
#         return line


# example = raw_data.filter(get_std_scaled).values().first()
# print(example)

# generate layout and plot data
# fig, ax = prepare_plot(np.arange(0, 300, 50), np.arange(-.1, .6, .1))
# ax.set_xlabel(r'time'), ax.set_ylabel(r'fluorescence')
# ax.set_xlim(-20, 260), ax.set_ylim(-.12, .52)
# plt.plot(range(len(example)), example, c='#8cbfd0', linewidth='3.0')
# plt.show()

"""
PCA on scaled data
-------------------

Dataset is preprocessed with n=46460 pixels and d=240 seconds of time series data for each pixel.
Interpret the pixels as our observations and each pixel value in the time series as a feature.
Find patterns in brain activity during this time series, its expected to find correlations over time.
The pca function takes an RDD of arrays, but scaled_data is an RDD of key-value pairs, extract only the values.
"""

# components_scaled, scaled_scores, eigenvalues_scaled = pca(scaled_data.map(lambda x: x[1]), k=3)

# scores_scaled = np.vstack(scaled_scores.collect())
# image_one_scaled = scores_scaled[:, 0].reshape(230, 202).T

# generate layout and plot data
# fig, ax = prepare_plot(np.arange(0, 10, 1), np.arange(0, 10, 1), figsize=(9.0, 7.2), hide_labels=True)
# ax.grid(False)
# ax.set_title('Top Principal Component', color='#888888')
# image = plt.imshow(image_one_scaled, interpolation='nearest', aspect='auto', cmap=cm.gray)
# fig.show()


"""
----------------------------------
Feature based aggregation and PCA
----------------------------------
"""


"""
Aggregation using arrays 
-------------------------

Earlier we performed PCA on the full time series data, trying to find global patterns across all 240 seconds of the time series.
But our analysis doesn't use the fact that different events happened during those 240 seconds.
Specifically, during those 240 seconds, the zebrafish was presented with 12 different direction-specific visual patterns, 
    with each one lasting for 20 seconds, for a total of 12 x 20 = 240 features.
We can isolate the impact of temporal response or direction-specific impact by appropriately aggregating our features.
"""

"""
Aggregate by time
--------------------

First study the temporal aspects of neural response, by aggregating our features by time.
In other words, we want to see how different pixels (and the underlying neurons captured in these pixels) react 
    in each of the 20 seconds after a new visual pattern is displayed, regardless of what the pattern is.
Instead of working with the 240 features individually, we'll aggregate the original features into 20 new features, 
    where the first new feature captures the pixel response one second after a visual pattern appears, 
    the second new feature is the response after two seconds, and so on.

We can perform this aggregation using a map operation.
First, build a multi-dimensional array 'T' that, when dotted with a 240-component vector, 
    sums every 20-th component of this vector and returns a 20-component vector.
After creating multi-dimensional array T, use a map operation with that array and each time series to generate a transformed dataset.
"""

# Create a multi-dimensional array to perform the aggregation
T = np.tile(np.eye(20), 12)

# Transform scaled_data using T.  Make sure to retain the keys.
time_data = scaled_data.map(lambda v: (v[0], np.dot(T, v[1])))
time_data.cache()

# print(time_data.count())
# print(time_data.first())
# print(T.shape)


"""
Obtain a compact representation
-------------------------------

We now have a time-aggregated dataset with n=46460 pixels and d=20 aggregated time features, 
    and we want to use PCA to find a more compact representation.
"""

components_time, time_scores, eigenvalues_time = pca(time_data.map(lambda x: x[1]), k=3)

# print('components_time: (first five) \n{0}'.format(components_time[:5, :]))
# print('\ntime_scores (first three): \n{0}'.format('\n'.join(map(str, time_scores.take(3)))))
# print('\neigenvalues_time: (first five) \n{0}'.format(eigenvalues_time[:5]))


"""
Aggregate by direction
----------------------

We want to see how different pixels (and the underlying neurons captured in these pixels) react when the 
    zebrafish is presented with 12 direction-specific patterns, ignoring the temporal aspect of the reaction.
Instead of working with the 240 features individually, we'll aggregate the original features into 12 new features, 
    where the first new feature captures the average pixel response to the first direction-specific visual pattern, 
    the second new feature is the response to the second direction-specific visual pattern, and so on.

Design a multi-dimensional array D that, when multiplied by a 240-dimensional vector, sums the first 20 components, 
    then the second 20 components, and so on
Create D, then use a map operation with that array and each time series to generate a transformed dataset.
"""

# Create a multi-dimensional array to perform the aggregation
D = np.kron(np.eye(12), np.ones(20))

# Transform scaled_data using D.  Make sure to retain the keys.
direction_data = scaled_data.map(lambda v: (v[0], np.dot(D, v[1])))

direction_data.cache()
# print(direction_data.count())
# print(direction_data.first())
# print(D.shape)

"""
Compact representation of direction data
----------------------------------------

We have a direction-aggregated dataset with n=46460 pixels and d=12 aggregated direction features, 
    and we want to use PCA to find a more compact representation.
"""

components_direction, direction_scores, eigenvalues_direction = pca(direction_data.map(lambda x: x[1]), k=3)

# print('components_direction: (first five) \n{0}'.format(components_direction[:5, :]))
# print('\ndirection_scores (first three): \n{0}'.format('\n'.join(map(str, direction_scores.take(3)))))
# print('\neigenvalues_direction: (first five) \n{0}'.format(eigenvalues_direction[:5]))




