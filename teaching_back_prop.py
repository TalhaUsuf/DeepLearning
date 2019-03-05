# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 00:33:33 2019

@author: Talha
"""

import numpy as np 


def relu(x):
    return x*(x>0)

m=10  # 10 examples
nl0=3 # input layer units
nl1=6 # ist hidden layer units
nl2=4 # 2nd hidden layer units
nl3=5 # no. of output layer units 

#       Here X is arbitrary image matrix
X=np.random.randint(low=0,high=256,size=(nl0,m)) # X shape = (3, 10), 10 examples, 3 attributes of image
#%%
def initialize(nl0,nl1,nl2,nl3):
    """
    Initialize parameters:
        nl0: 
            no. of units in input layer
        nl1: 
            no. of units in h1 layer
        nl2: 
            no. of inputs in h2 layer
        nl3: 
            no. of inputs in output layer
    Output:
        A dictionary of W's and b's

    """

    W1=np.random.randn(nl1,nl0)*0.01 # W1 shape = (6, 3)
    W2=np.random.randn(nl2,nl1)*0.01 # W2 shape = (4, 6)
    W3=np.random.randn(nl3,nl2)*0.01 # W3 shape = (5, 4)
    
    b1=np.zeros(shape=(nl1,1)) # b1 shape = (6, 1)
    b2=np.zeros(shape=(nl2,1)) # b2 shape = (4, 1)
    b3=np.zeros(shape=(nl3,1)) # b3 shape = (5, 1)
    
    parameters={'W1':W1,
                'W2':W2,
                'W3':W3,
                'b1':b1,
                'b2':b2,
                'b3':b3}
    return parameters

def forward_propagate(parameters,X):
    """
    forward_propagate(weights , input_images_matrix)
    
    Forward Propagation to compute scores of activations.
    
    Parameters
    ----------
    parameters : dictionary of size 6
        It should hold values of weights and biases in form of dictionary
    X          : Input images 
        `X` should be matrix of input images where each column is an example.
        Shape of `X` should be (nl0,m)
    """
    W1=parameters['W1']
    W2=parameters['W2']
    W3=parameters['W3']
    b1=parameters['b1']
    b2=parameters['b2']
    b3=parameters['b3']
    
    Z1=np.add(np.dot(W1,X) , b1) # Z1 shape = (6, 10) --> (nl1,m)
    A1 = relu(Z1)
    Z2=np.add(np.dot(W2,A1), b2) # Z2 shape = (4, 10) --> (nl2,m)
    A2= relu(Z2)
    Z3=np.add(np.dot(W3,A2),b3)  # Z3 shape = (5, 10) --> (nl3,m)
    A3 = relu(Z3)
    
    scores={'Z1':Z1,
            'A1':A1,
            'Z2':Z2,
            'A2':A2,
            'Z3':Z3,
            'A3':A3}
    return scores
# =============================================================================
#                   Calling functions
#                   Now do back propagation in same manner
# =============================================================================
par=initialize(3,10,15,5) 
scores=forward_propagate(par,X)
