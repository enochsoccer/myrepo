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

def E_Regular(weights,phi,yn):
    # Computes the in-sample or out-of-sample error for the regular form RBF model, given weight values
    #   (including bias value b), the 'phi' matrix (expected to have a first column of 1's), and 'yn' labels.
    b = weights[0]
    w = weights[1:]
    N = len(phi[:,0])
    bmat = np.repeat(b,N).reshape(-1,1)  # converts to column vector
    yest = np.sign(phi[:,1:]@w + bmat)  # estimated labels; removed first column of 1's
    E = np.sum(yest != yn)/N
    return E

def E_Kernel(svmMod,xn,yn,gamma):
    # Computes the in-sample or out-of-sample error for the kernel form RBF model. Inputs are the SVM model 'svmModel',
    #   input points 'xn', labels 'yn', and the 'gamma' value.
    b = svmMod.intercept_
    N = len(yn)
    bmat = np.repeat(b,N).reshape(-1,1)  # converts to column vector
    coefs = svmMod.dual_coef_.reshape(-1,1)  # converts to column vector
    SVs = svmMod.support_vectors_
    phi = phiMat(xn,SVs,gamma)  # where originally 'mus' go in, I've replaced it with 'SVs'
    yest = np.sign(phi@coefs + bmat)  # estimated labels as a result of training
    E = np.sum(yest != yn) / N
    return E

def RBFregularform(K,xn,mus,yn,g,pl):
    # Computes the RBF model in the regular form.
    N = len(yn)
    if pl: plot(xn, mus, color='red')
    while True:
        # Iteration while-loop
        bins = assignBins(xn, mus, K)  # assign a mu to each xn, i.e., bins 1 to K for each xn
        if emptyBins(bins, K):  # if there is an empty cluster/bin, then redo run
            # Generate new data and random center mu's
            #TESTING: xn = generatexn(N)
            #TESTING: yn = generateyn(xn)
            #TESTING: xn, yn = checkyn(xn, yn)
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
    return weights, mus

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
problem = 18

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
        Ein = E_Kernel(model, xn, yn, g)
        if Ein: count += 1  # i.e., if data is linearly inseparable
    print(count / runs)

elif problem == 14 or problem == 15:
    # Discard runs if:
    #    (1) For RBF kernel: (Ein != 0) i.e., data is not separable in the Z-space by the RBF kernel w/ hard-margin SVM
    #    (2) For RBF regular: if a cluster becomes empty
    K = 12  # K-centers
    N = 100  # number of training points
    N_out = 100  # number of out-of-sample points
    g = 1.5  # gamma
    runs = 10000
    count_invalid = 0  # number of invalid runs
    E_outs_k = np.empty(runs)
    E_outs_r = np.empty(runs)
    pl = False  # plot?

    for i in range(runs):
        # Training both regular and kernel RBF forms
        xn = generatexn(N)
        yn = generateyn(xn)
        xn, yn = checkyn(xn, yn)
        # RBF regular form
        mus = np.random.uniform(-1, 1, (K,2))  # initiate random center mu's
        weights, mus = RBFregularform(K, xn, mus, yn, g, pl)
        # RBF kernel form
        model = RBFkernelform(xn, yn, g)  # majority times, expecting Ein = 0 (i.e., linearly separable)
        Ein = E_Kernel(model, xn, yn, g)  # checking to see if linearly separable
        if Ein > 0:
            count_invalid += 1
            continue  # returns to beginning of for-loop

        # Calculating out-of-sample errors
        xn_out = generatexn(N_out)
        yn_out = generateyn(xn_out)
        E_outs_k[i] = E_Kernel(model, xn_out, yn_out, g)  # kernel form
        phi_out = phiMat(xn_out, mus, g)
        phi_out = np.concatenate((np.ones((N_out,1)), phi_out), axis=1)  # adding column of 1's to compute for the bias term in the next line
        E_outs_r[i] = E_Regular(weights, phi_out, yn_out)  # regular form

    # Comparing how often kernel form beats regular form
    ratio = sum(E_outs_k < E_outs_r) / (runs - count_invalid)
    print(count_invalid, "invalid counts out of", runs, "runs.")
    print("Kernel form beats regular form %:", ratio*100)

elif problem == 16:
    K = 12  # K-centers
    N = 100  # number of training points
    N_out = 100  # number of out-of-sample points
    g = 1.5  # gamma
    runs = 7500
    E_ins_r = np.empty(runs)
    E_outs_r = np.empty(runs)
    pl = False  # plot?

    for i in range(runs):
        # Training - RBF regular form
        xn = generatexn(N)
        yn = generateyn(xn)
        xn, yn = checkyn(xn, yn)
        mus = np.random.uniform(-1, 1, (K,2))  # initiate random center mu's
        weights, mus = RBFregularform(K, xn, mus, yn, g, pl)
        phi = phiMat(xn, mus, g)
        phi = np.concatenate((np.ones((N,1)), phi), axis=1)  # adding column of 1's to compute for the bias term in t
        E_ins_r[i] = E_Regular(weights, phi, yn)

        # Calculating out-of-sample errors
        xn_out = generatexn(N_out)
        yn_out = generateyn(xn_out)
        phi_out = phiMat(xn_out, mus, g)
        phi_out = np.concatenate((np.ones((N_out,1)), phi_out), axis=1)  # adding column of 1's to compute for the bias term in the next line
        E_outs_r[i] = E_Regular(weights, phi_out, yn_out)

    print("Avg Ein =", np.mean(E_ins_r))
    print("Avg Eout =", np.mean(E_outs_r))

elif problem == 17:
    K = 9  # K-centers
    N = 100  # number of training points
    N_out = 100  # number of out-of-sample points
    gammas = [1.5, 2]  # gamma values
    runs = 2000
    E_vals_g = np.empty((len(gammas),2))  # first col = avg Ein; second col = avg Eout
    E_ins_r = np.empty(runs)  # RBF regular form
    E_outs_r = np.empty(runs)
    pl = False  # plot?

    for i_g, g in enumerate(gammas):

        for i in range(runs):
            # Training - RBF regular form
            xn = generatexn(N)
            yn = generateyn(xn)
            xn, yn = checkyn(xn, yn)
            mus = np.random.uniform(-1, 1, (K,2))  # initiate random center mu's
            weights, mus = RBFregularform(K, xn, mus, yn, g, pl)
            phi = phiMat(xn, mus, g)
            phi = np.concatenate((np.ones((N,1)), phi), axis=1)  # adding column of 1's to compute for the bias term in t
            E_ins_r[i] = E_Regular(weights, phi, yn)

            # Calculating out-of-sample errors
            xn_out = generatexn(N_out)
            yn_out = generateyn(xn_out)
            phi_out = phiMat(xn_out, mus, g)
            phi_out = np.concatenate((np.ones((N_out,1)), phi_out), axis=1)  # adding column of 1's to compute for the bias term in the next line
            E_outs_r[i] = E_Regular(weights, phi_out, yn_out)

        E_vals_g[i_g,0] = np.average(E_ins_r)  # first col = avg Ein
        E_vals_g[i_g,1] = np.average(E_outs_r)  # second col = avg Eout
        print("Gamma:", g, "| E_in =", E_vals_g[i_g,0], ", E_out =", E_vals_g[i_g,1])

elif problem == 18:
    K = 9  # K-centers
    N = 100  # number of training points
    g = 1.5  # gamma
    runs = 2500
    E_ins_r = np.empty(runs)  # RBF regular form
    pl = False  # plot?

    for i in range(runs):
        # Training - RBF regular form
        xn = generatexn(N)
        yn = generateyn(xn)
        xn, yn = checkyn(xn, yn)
        mus = np.random.uniform(-1, 1, (K,2))  # initiate random center mu's
        weights, mus = RBFregularform(K, xn, mus, yn, g, pl)
        phi = phiMat(xn, mus, g)
        phi = np.concatenate((np.ones((N,1)), phi), axis=1)  # adding column of 1's to compute for the bias term in t
        E_ins_r[i] = E_Regular(weights, phi, yn)

    ratio = sum(E_ins_r == 0) / runs
    print(ratio*100, "% of runs achieved E_in = 0")