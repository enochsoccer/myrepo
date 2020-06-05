'''
Machine Learning
HW #5, Problems 8-10
Assignment: http://work.caltech.edu/homework/hw5.pdf
'''

import numpy as np
import sys # for testing purposes sys.exit()

# Functions
def generatexn(N):
    # Generates N input points xn of dimension: (1,x1,x2)^T x N and returns
    #   the transposed version N x (x0,x1,x2) for easier iteration
    xn = np.random.uniform(-1,1,(2,N))
    xn = np.insert(xn,0,np.ones(N),axis=0)
    xn = np.transpose(xn)
    return xn # [[1 x1 x2]
              #  [1 x1 x2]
              #     ...
              #  [1 x1 x2]]

def generateTargetFeu():
    # Generates target function f
    vals = np.random.uniform(-1,1,4)
    (x1,x2,y1,y2) = vals
    m = (y2-y1)/(x2-x1)
    return(m,x1,y1)

def applyTargetFeu(a,b,m,x1,y1):
    # Generates target value yn for input point (a,b), given the target feu
    #   described by values m, x1, and y1
    yref = m*(a-x1) + y1
    if b > yref:
        return +1
    else:
        return -1

def generateyn(xn,m,x1,y1):
    # Generates all yn target values given all the input xn values and the
    #   target feu characterized by (m,x1,y1) values
    N = len(xn)
    yn = np.empty(N)
    for i in range(N):
        yn[i] = applyTargetFeu(xn[i][1],xn[i][2],m,x1,y1)
    return yn

def logisticRegAlg(xn,yn,N,eta):
    # Computes the Logistic Regression algorithm using stochastic gradient descent (SGD).
    #   SGD iterates through one example point at a time. Each Regression experiment
    #   goes in a random permutation of indices from 'indexList'
    w = np.array([0,0,0])
    epoch = 0
    while 1:
        indexList = np.random.permutation(N) # randomly permutes 0,1,...,(N-1)
        wprev = w
        gradient = np.array([0,0,0])
        for i in indexList: # Iterates through all N points in xn for the gradient
            x = xn[i]
            y = yn[i]
            constant = y / ( 1 + np.exp(y*np.dot(w,x)) ) / (-N)
            gradient = gradient + constant*x
            w = w - eta*gradient
        epoch = epoch + 1
        if np.linalg.norm(wprev - w) < 0.01:
            break
    return (w,epoch)

def Eout(Nout,w,m,x1,y1):
    # Computes the out-of-sample error on a separate set of Nout points.
    xn = generatexn(Nout)
    yn = generateyn(xn,m,x1,y1)
    for i in range(Nout):
        s = np.log( 1 + np.exp(-yn[i]*np.dot(w,xn[i])) )
    return (s/Nout)

# Parameters
N = 100 # Training points
Nout = 1000 # Out-of-sample points
eta = 0.01 # Learning rate
runs = 100

# Processing
xn = generatexn(N) # generate data
wVals = np.empty([runs,3]) # weights w from hypothesis g
eVals = np.empty(runs) # epoch values
EoutVals = np.empty(runs) # Eout values
for i in range(runs):
    (m,x1,y1) = generateTargetFeu()
    yn = generateyn(xn,m,x1,y1)
    (w,epoch) = logisticRegAlg(xn,yn,N,eta) # Generates hypothesis g with weights w
    wVals[i][:] = w
    eVals[i] = epoch
    EoutVals[i] = Eout(Nout,w,m,x1,y1)

# Computing the run averages
wVals_avg = wVals.mean(axis=0)
eVals_avg = eVals.mean(axis=0)
EoutVals_avg = EoutVals.mean(axis=0)

# Display results
print('avg weight =',wVals_avg)
print('avg epoch =',eVals_avg)
print('avg Eout =',EoutVals_avg)