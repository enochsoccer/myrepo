"""
Machine Learning, Final Assignment, Problems #7-10
"""

import numpy as np
import copy

# Functions
def datagrab(which):
    # Grabs either "training" or "test" data. "which" should be a string.
    name = "final-" + which + ".txt"
    data = np.loadtxt(name)
    return data

def rawdata():
    # Returns the raw (i.e., digit, intensity, symmetry) training and test data. Dim = Nx3
    training = datagrab("training")
    test = datagrab("test")
    return (training,test)

def oneVall(dig,data):
    # "dig" is class +1 and the rest of the digits are class −1. "data" is the full data set.
    #   Returns "xn", with digits replaced by 1s, and "yn" labels.
    digits = data[:,0]
    idxes = np.where(digits == dig)[0]
    N = data.shape[0]
    yn = -1*np.ones(N)  # first fill with all -1 labels
    yn[idxes] = 1  # then replace appropriate indices with +1 labels
    yn = yn.reshape(-1,1)  # turn into column vector
    xn = copy.deepcopy(data)  # copy the data set and then...
    xn[:,0] = 1  # replace digits with bias values of 1
    return (xn,yn)  # xn = Nx3 and yn = Nx1

def oneVone(dig1,dig2,data):
    # "dig1" is class +1 and "dig2" is class −1, with the rest of the digits disregarded.
    #   "data" is the full data set. Returns "xn", with digits replaced by 1s, and its labels "yn".
    digits = data[:,0]
    dig1_i = np.where(digits == dig1)[0]  # find indices of "dig1" occurrences
    dig2_i = np.where(digits == dig2)[0]  # same for "dig2"
    idxes = np.sort(np.concatenate([dig1_i, dig2_i]), axis=None)  # sort the indices in increasing order
    data_red = data[idxes,:]  # non-relevant digits disregarded (hence data set reduced)

    # Working with reduced data set.
    N = data_red.shape[0]
    yn = np.ones(N)  # first fill with all +1 labels
    digits = data_red[:,0]
    minus_i = np.where(digits == dig2)[0]  # find indices for -1 labels
    yn[minus_i] = -1  # replace appropriate indices with -1 labels
    yn = yn.reshape(-1,1)  # turn into column vector
    xn = copy.deepcopy(data_red)  # copy the reduced data set and then...
    xn[:,0] = 1  # replace digits with bias values of 1
    return(xn,yn)  # xn = Nx3 and yn = Nx1 *** double check these dimensions!!!!!!!

def wreg(xn,yn,lam):
    # Computes the weight vector with regularization.
    XT = xn.transpose()  # Also works with the Z transform matrix
    X = xn
    N = X.shape[1]
    I = np.identity(N)
    w = (np.linalg.inv(XT@X + lam*I) @ XT) @ yn  # @ stands for matrix multiplication
    return w

def h(w,x):
    # Returns hypothesis value of single data point "x" = (1,x1,x2)^T, given weight "w" = (w0,w1,w2)^T.
    wT = w.reshape(1, -1)  # turn into row vector
    return np.sign(np.dot(wT,x))

def E(w,xn,yn):
    # Computes binary error for data "xn" and its labels "yn". Target function f(x) values are "yn".
    N = len(yn)
    hX = np.empty((N,1))
    for i in range(N):
        hX[i] = h(w,xn[i,:])
    return np.sum(hX != yn) / N

def ztransf(xn):
    # Applies the z-transform = (1, x1, x2, x1*x2, x1^2, x2^2) on "xn" = Nx3 to return "zn" = Nx6.
    N = xn.shape[0]
    zn = copy.deepcopy(xn)
    new = np.empty((N,3))  # add three new columns
    zn = np.append(zn,new,axis=1)  # expands to Nx6 matrix
    zn[:,3] = np.multiply(xn[:,1],xn[:,2])  # x1*x2
    zn[:,4] = np.square(xn[:,1])
    zn[:,5] = np.square(xn[:,2])
    return zn

def K(a,b):
    # Computes the second-order kernel polynomial, K(x,x') = (1 + xT x')^2,
    #   given inputs "a" and "b" (equivalent of x and x'). Assumes "a" and "b" = Nx1.
    aT = a.reshape(1, -1)  # turn into row vector
    return (1 + aT@b) ** 2  # @ stands for matrix multiplication, ** is for the exponent

def quadMat(xn,yn):
    # Computes the quadratic matrix coefficients with the kernel. Expects dim(xn) = 7x2
    N = len(yn)
    mat = np.empty((N,N))
    for i in range(N):
        for j in range(N):
            mat[i, j] = yn[i] * yn[j] * K(xn[i, :], xn[j, :])
    return mat

# ------------------------------------------------------------------------------

(train_raw, test_raw) = rawdata()
prob = 12

# Processing
if prob == 7:
    digits = [5, 6, 7, 8, 9]
    lam = 1  # lambda value
    Ein_vals = np.empty(len(digits))
    for i in range(len(digits)):
        # In-sample training and testing
        (xn_tr, yn_tr) = oneVall(digits[i], train_raw)
        w = wreg(xn_tr, yn_tr, lam)
        Ein_vals[i] = E(w, xn_tr, yn_tr)
    print(Ein_vals)
elif prob == 8:
    digits = [0, 1, 2, 3, 4]
    lam = 1  # lambda value
    Eout_vals = np.empty(len(digits))
    for i in range(len(digits)):
        # In-sample training
        (xn_tr, yn_tr) = oneVall(digits[i], train_raw)
        zn_tr = ztransf(xn_tr)
        w = wreg(zn_tr, yn_tr, lam)

        # Out-of-sample testing
        (xn_test, yn_test) = oneVall(digits[i], test_raw)
        zn_test = ztransf(xn_test)
        Eout_vals[i] = E(w, zn_test, yn_test)
    print(Eout_vals)
elif prob == 9:
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    lam = 1  # lambda value
    Eout_vals = np.empty(len(digits))  # without using transforms
    Eout_tr_vals = np.empty(len(digits))  # using transforms
    for i in range(len(digits)):
        # In-sample training
        (xn_tr, yn_tr) = oneVall(digits[i], train_raw)
        zn_tr = ztransf(xn_tr)  # non-linear transform matrix
        w = wreg(xn_tr, yn_tr, lam)
        w_trans = wreg(zn_tr, yn_tr, lam)  # weight for non-linear transformation

        # Out-of-sample testing
        (xn_test, yn_test) = oneVall(digits[i], test_raw)
        zn_test = ztransf(xn_test)  # non-linear transform matrix
        Eout_vals[i] = E(w, xn_test, yn_test)
        Eout_tr_vals[i] = E(w_trans, zn_test, yn_test)  # Eout for non-linear transformation
    print(Eout_vals)
    print(Eout_tr_vals)
elif prob == 10:
    dig1 = 1
    dig2 = 5
    lambdas = [0.01, 1]
    Ein_vals = np.empty(len(lambdas))
    Eout_vals = np.empty(len(lambdas))
    for i in range(len(lambdas)):
        # In-sample training and test
        (xn_tr, yn_tr) = oneVone(dig1, dig2, train_raw)
        zn_tr = ztransf(xn_tr)  # non-linear transform matrix
        w_trans = wreg(zn_tr, yn_tr, lambdas[i])  # weight for non-linear transformation
        Ein_vals[i] = E(w_trans, zn_tr, yn_tr)
        
        # Out-of-sample testing
        (xn_test, yn_test) = oneVone(dig1,dig2,test_raw)
        zn_test = ztransf(xn_test)  # non-linear transform matrix
        Eout_vals[i] = E(w_trans, zn_test, yn_test)  # Eout for non-linear transformation
    print(Ein_vals)
    print(Eout_vals)
elif prob == 12:
    """
    # I couldn't figure out how to resolve implementation issues of cvxopt package. Solved this problem in R.
    
    xn = np.array([[-1, 0], [0, 1], [0, -1], [1, 0], [0, 2], [0, -2], [-2, 0]])  # switched x1 and x4 in prob statement
    yn = np.array([1, -1, -1, -1, 1, 1, 1])  # switched x1 and x4 in prob statement to ensure first entry is +1
    N = len(yn)
    
    G = np.double( quadMat(xn,yn) )
    a = np.double( np.ones(N) )
    C = copy.deepcopy(yn)
    C = np.double( C.reshape(-1, 1) )  # turn into a column vector
    b = np.double( 0 )
    meq = N  # makes it such that all constraints are equality constraints
    qp = quadprog.solve_qp(G, a, C, b, meq)  # doc: https://github.com/rmcgibbo/quadprog/blob/master/quadprog/quadprog.pyx
    """

    """
    P = matrix( quadMat(xn,yn) )
    q = matrix( -1, (1,N) )
    G = matrix(0.0, (N,N))
    G[::N+1] = -1.0
    h = matrix(0., (N,1))
    A = matrix( yn, (1,N) )
    b = matrix(0.)
    z = cvxopt.solvers.qp(P, q, G, h, A, b)"""

    P = quadMat(xn, yn)
    q = matrix(-1, (1, N))
    G = matrix(0.0, (N, N))
    G[::N + 1] = -1.0
    h = matrix(0., (N, 1))
    A = matrix(yn, (1, N))
    b = matrix(0.)
    z = solvers.qp(P, q, G, h, A, b)