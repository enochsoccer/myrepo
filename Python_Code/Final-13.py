"""
Machine Learning, Final Assignment, Problem #13
"""

import numpy as np
import copy
from matplotlib import pyplot as plt
from sklearn.svm import SVC
import sys  # for testing purposes

def generatexn(N):
    # Generates N input (x1,x2) points xn of dimension: N x 2 and returns
    xn = np.random.uniform(-1,1,(N,2))
    return xn # [[x1 x2]
              #  [x1 x2]
              #     ...
              #  [x1 x2]

def f(x1,x2):
    # Returns the target function value f(x) = sign(x2 - x1 + 0.25sin(pi*x1))
    return np.sign( x2 - x1 + 0.25*np.sin(np.pi*x1) )

def generateyn(xn):
    # Generates all yn target values given all the input xn values of dim = N x 2.
    N = xn.shape[0]
    yn = np.empty(N)
    for i in range(N):
        yn[i] = f(xn[i,0], xn[i,1])
    yn = yn.reshape(-1, 1)  # column vector
    return yn

def distSq(x,y):
    # Computes the squared Euclidean distance value between points x and y.
    return (x[0]-y[0])**2 + (x[1]-y[1])**2

def assignBins(xn,mus,K):
    # Assigns a number from K bins to each data point in xn. "mus" holds K centers.
    #   dim(xn) = N x 2
    #   dim(mus) = K x 2

    # Assign K distance values for each data point in xn.
    N = xn.shape[0]
    distances = np.zeros((N,K))  # row = xn data point, col = distance value for center K
    bins = np.empty(N)  # assigned bin number for each xn points
    for i in range(N):
        # Assigning K distances to xn[i,:]
        for j in range(K):
            distances[i,j] = distSq(xn[i,:],mus[j,:])
        # Finding the closest center mu_k.
        idx = np.argmin(distances[i,:])
        bins[i] = idx + 1
    return bins

def emptyBins(bins,K):
    # Returns true if there is an empty bin from bins 1 to K.
    unq = np.unique(bins)  # returns np array of sorted unique elements in "bins"
    return True if len(unq) < K else False

def newMu(xn):
    # Receives data points from *one* cluster and computes the centroid of these points.
    #   Assumes dim(xn) = N x 2. Returns (x,y) coordinates in "mu".
    mu = np.average(xn, axis=0)  # axis=0 averages over columns
    return mu

def newCenters(xn,bins,K,mus):
    # This function acts as the iteration component of Lloyd's algorithm. Computes and returns
    #   the new centers for all clusters.
    for i in range(K):  # Iterates through one bin (or cluster) at a time
        binNo = i + 1  # Bin number
        clusterPts = xn[ np.where(bins == binNo)[0], : ]  # Grabs all points in a single cluster.
        mus[i,:] = newMu(clusterPts)  # Computes new centroid based off cluster points.
    return mus

def phiMat(xn,mus,gamma):
    # Returns the phi matrix composed of exp(-gamma * ||x - mu||^2) that will be used for
    #   computing the weights from Lloyd's algorithm. dim(phi) = N x K
    N = xn.shape[0]
    K = mus.shape[0]
    phi = np.zeros((N,K))
    for i in range(N):
        for j in range(K):
            phi[i,j] = np.exp( -gamma * distSq(xn[i,:], mus[j,:]) )
    return phi

def plot(xn,mus=None,color=None):
    # Plots the data points and centers.
    if mus is None:
        plt.plot(xn[:, 0], xn[:, 1], 'x', c=color)
    else:
        plt.xlabel("x_1")
        plt.ylabel("x_2")
        plt.plot(xn[:, 0], xn[:, 1], 'o')
        plt.plot(mus[:, 0], mus[:, 1], 'x', c=color)

def swap(arr,i,j):
    # Swaps the array values.
    try:
        arr.shape[1]  # tests if array is two-dimensional
        temp = np.copy(arr[j,:])
        arr[j,:] = np.copy(arr[i,:])
        arr[i,:] = temp
    except IndexError:
        temp = np.copy(arr[j])
        arr[j] = np.copy(arr[i])
        arr[i] = temp
    return arr

def checkyn(xn,yn):
    # Checks to see if the first yn has a -1 label. If so, move the data around so that first label is +1.
    if yn[0] == -1:
        i = np.where(yn == +1)[0][0]  # first instance of +1 label
        yn = swap(yn,0,i)
        xn = swap(xn,0,i)
    return xn, yn

def EinRegular(weights,phi,yn):
    # Computes the in-sample training error for the regular form RBF model, given weight values (including bias value b)
    #   and 'phi' matrix, which includes the xn and mu data inherently. 'phi' is expected to have a first column of 1's.
    b = weights[0]
    w = weights[1:]
    N = len(phi[:,0])
    bmat = np.repeat(b,N)
    bmat = bmat.reshape(-1,1)  # converts to column vector
    yest = np.sign(phi[:,1:]@w + bmat)  # estimated labels as a result of training; removed first column of 1's
    Ein = np.sum(yest != yn)/len(yn)
    return Ein

def EinKernel(svmMod,xn,yn,gamma):
    # Computes the in-sample training error for the kernel form RBF model. Inputs are the SVM model 'svmMod',
    #   input points 'xn', and training labels 'yn'.
    b = svmMod.intercept_
    N = len(yn)
    bmat = np.repeat(b,N).reshape(-1,1)  # converts to column vector
    coefs = svmMod.dual_coef_.reshape(-1,1)  # converts to column vector
    SVs = svmMod.support_vectors_
    phi = phiMat(xn,SVs,gamma)  # where originally 'mus' go in, I've replaced it with 'SVs'
    yest = np.sign(phi@coefs + bmat)  # estimated labels as a result of training
    Ein = np.sum(yest != yn) / N
    return Ein

def RBFregularform(K,xn,mus,yn,g,pl):
    # Computes the RBF model in the regular form.
    N = len(yn)
    if pl: plot(xn, mus, color='red')
    while True:
        # Iteration while-loop
        bins = assignBins(xn, mus, K)  # assign a mu to each xn, i.e., bins 1 to K for each xn
        if emptyBins(bins, K):  # if there is an empty cluster/bin, then redo run
            # Generate new data and random center mu's
            xn = generatexn(N)
            yn = generateyn(xn)
            xn, yn = checkyn(xn, yn)
            mus = np.random.uniform(-1, 1, (K, 2))
        else:  # if none of the bins are empty, continue Lloyd's algorithm
            prevMus = copy.deepcopy(mus)  # memorize the previous centers
            mus = newCenters(xn, bins, K, mus)  # compute the new centers  # HAVEN'T TESTED BEYOND THIS LINE, INCLUSIVE
            if np.all(mus == prevMus):
                if pl: plot(mus, color='purple')
                plt.show()
                break  # end of training
            if pl: plot(mus, color='orange')
    phi = phiMat(xn, mus, g)
    phi = np.concatenate((np.ones((N, 1)), phi), axis=1)  # adding column of 1's to compute for the bias term in the next line
    weights = (np.linalg.inv(phi.T @ phi)) @ phi.T @ yn  # pseudo-inverse; includes bias term b
    return weights, phi

def RBFkernelform(xn,yn,g):
    # Computes the RBF model in the kernel form.
    clf = SVC(C=1e9, kernel='rbf', gamma=g)  # classifier
    svmModel = clf.fit(xn,np.ravel(yn))  # np.ravel just flattens the column vector to a row vector. scikit preference.
    return svmModel

def statistics(vals,runs):
    # This method provides insights on whether your answer is statistically stable, i.e., statistically
    #   away from flipping to the closest competing answer. http://book.caltech.edu/bookforum/showthread.php?t=4333
    sigma = np.std(vals)
    stat = sigma/np.sqrt(runs)  # represents standard deviation per run (I think)
    print("Your value is",stat,"standard deviations away flipping to another answer.")
    return


# Which problem?
problem = 13

if problem == 13:
    # RBF model - kernel form
    N = 100  # number of training points
    g = 1.5  # gamma
    runs = 20000
    count = 0  # count how many times data is linearly inseparable in Z-space
    for i in range(runs):
        xn = generatexn(N)
        yn = generateyn(xn)
        xn, yn = checkyn(xn, yn)
        model = RBFkernelform(xn, yn, g)  # majority times, expecting Ein = 0 (i.e., linearly separable)
        Ein = EinKernel(model, xn, yn, g)
        if Ein: count += 1  # i.e., if data is linearly inseparable
    print(count / runs)

elif problem == 14:
    # THIS IS WHERE YOU LEFT OFF. REMEMBER THE DETAIL ABOUT DISCARDING RUNS, MENTIONED IN PROBLEM 14.


# RBF model - regular form
K = 9  # K-centers
N = 100  # number of training points
g = 1.5  # gamma
mus = np.random.uniform(-1, 1, (K,2))  # initiate random center mu's
xn = generatexn(N)
yn = generateyn(xn)
xn, yn = checkyn(xn,yn)
pl = False  # plot?
weights, phi = RBFregularform(K,xn,mus,yn,g,pl)
Ein = EinRegular(weights,phi,yn)
print(Ein)






