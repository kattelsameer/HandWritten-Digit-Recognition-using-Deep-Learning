#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:41:53 2020

@author: befrenz
"""
import numpy as np
#import matplotlib.pyplot as plt
#---------------------------------------------------------------------------------------------------------------
#relu function
def relu(Z):
    """
        Compute the ReLU activation of Z
        
        Argument:
            - Z -- Array of the Sum of the product of Weights and input
        
        Returns:
            - A -- Array of Activation obtained by applying ReLU function. same size as that of Z
    """
    A = np.maximum(0.0,Z)
    
    cache = Z
    assert(A.shape == Z.shape)
    return A, cache

#---------------------------------------------------------------------------------------------------------------
#relu gradient function
def relu_grad(dA, cache):
    """
        Compute the gradient of dA
        
        Arguments:
            - dA -- Array of the gradient of activation of the previous layer
            - cache -- list of other useful variables like Z
            
        Returns:
            - dZ -- array of gradient/derivative of the dA, Same size of dA
    """
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    dZ[Z < 0] = 0
    
    assert(dZ.shape == Z.shape)
    return dZ

#---------------------------------------------------------------------------------------------------------------
#softmax function
def softmax(Z):
    """
        Compute the softmax activtion of Z
        
        Argument:
            - Z -- Array of the Sum of the product of Weights and input
        
        Returns:
            - A -- Array of Activation obtained by applying Softmax function. same size as that of Z
    """
    shift = Z - np.max(Z) #Avoiding underflow or overflow errors due to floating point instability in softmax
    t = np.exp(shift)
    A = np.divide(t,np.sum(t,axis = 0))
    
    cache = Z
    assert(A.shape == Z.shape)
    return A, cache