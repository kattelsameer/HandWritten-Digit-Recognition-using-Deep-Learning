#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:26:57 2020

@author: befrenz
"""
import time    #for calculating time
import os

#core packages
import numpy as np

#custom modules
from ModelUtils import relu, relu_grad, softmax
from ModelUtils import rand_mini_batches, convert_time
from ModelUtils import save_model, load_model

from dataAugmentation import data_generator

#====================================================================================================================
# initializing the layers
def init_layers(input_shape, output_shape, hidden_layers):
    """Initializes the layers of networks with the number of nodes in each layers.
        
        Arguments:
            mnjnj.
            
        Returns:
            knlknl.
            
        Example:
            >>> layers_dim = init_layers(784, 10, hidden_layers = [32,16])
            >>> print(layers_dim)
            
            Outputs:
                [784, 32, 16, 10]
    """
    input_nodes = input_shape
    output_nodes = output_shape
    
    layers_dim = [input_nodes]
    
    for i in hidden_layers:
        layers_dim.append(i)
    
    layers_dim.append(output_nodes)
    
    return layers_dim

#====================================================================================================================
# initializing parameters
def init_parameters(layers_dim, initialization = "random"):
    """Initializes the parameters (W,b) for each layer.
        
        Arguments:
            kngnjgjg.
            
        Returns:
            nkngkg.
            
        Example:
            Here, layers_dim = [784, 32, 16, 10]
            >>> parameters = init_parameters(layers_dim, initialization = "random")
            >>> print("Layer\tWeight\t\tBias")
            >>> print("================================")
            >>> for l in range(1,len(layers_dim)):
            ...     print(str(l) +"\t" + str(parameters['W'+str(l)].shape) +"\t"+ str(parameters['b'+str(l)].shape))

            
            Outputs:
                Layer    Weight         Bias
                ================================
                1        (32, 784)      (32, 1)
                2        (16, 32)       (16, 1)
                3        (10, 16)       (10, 1)
    """
    L = len(layers_dim)
    params = {}
        
    for l in range(1,L):
        #initializing Weights
        if initialization == "he":
            # he-initialization
            params['W' + str(l)] = np.random.randn(layers_dim[l],layers_dim[l-1]) * np.sqrt(np.divide(2,layers_dim[l-1])) 
        elif initialization == "random":
            # random initialization scaled by 0.01
            params['W' + str(l)] = np.random.randn(layers_dim[l],layers_dim[l-1]) * 0.01 
        else:
             raise ValueError("Initialization must be 'random' or 'he'")
        
        #initializing biases
        params['b' + str(l)] = np.zeros((layers_dim[l],1))
     
        assert(params['W' + str(l)].shape == (layers_dim[l],layers_dim[l-1])), "Dimention of W mismatched in init_params function"
        assert(params['b' + str(l)].shape == (layers_dim[l],1)), "Dimention of b mismatched in init_params function"
   
    return params

#====================================================================================================================
# initializing hyper parameters
def init_hyperParams(alpha, num_epoch, minibatch_size = 64, lambd = 0, keep_probs = [],beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    """
        
        Arguments:
            knbknc.
            
        Returns:
            getget.
            
        Example:
            >>> hyperParams = init_hyperParams(alpha = 0.0001, num_epoch = 10, minibatch_size = 1024,lambd = 0.7,keep_probs = [0.8,0.8])
    """
    hyperParams = {'learning_rate':alpha,
                   'num_epoch':num_epoch,
                   'mini_batch_size':minibatch_size,
                   'lambda':lambd,
                   'keep_probs':keep_probs,
                   'beta1':beta1,
                   'beta2':beta2,
                   'epsilon':epsilon
                  }
    
    return hyperParams

#====================================================================================================================
# Forward Propagation
#-------------------------------------------------------------------------------------------------------------------
## forward sum
def forward_sum(A_prev,W,b):
    """
    
    
        Example:
            >>> np.random.seed(1)
            >>> A = np.random.randn(3,2)
            >>> W = np.random.randn(1,3)
            >>> b = np.random.randn(1,1)
            >>> Z, c = forward_sum(A,W,b)
            >>> print("Z = "+ str(Z))
            
            Output:
                Z = [[ 3.26295337 -1.23429987]]
        
    """
    m = A_prev.shape[1]
    
    Z = np.dot(W,A_prev) + b
    
    cache = (A_prev,W,b)
    
    assert (Z.shape == (W.shape[0], m)), "Dimention of Z mismatched in forward_prop function"
    
    return Z, cache

#-------------------------------------------------------------------------------------------------------------------
## forward Activation
def forward_activation(A_prev,W,b,activation):
    """
    
    
        Example:
            >>> np.random.seed(1)
            >>> A_prev = np.random.randn(3,2)
            >>> W = np.random.randn(1,3)
            >>> b = np.random.randn(1,1)

            >>> A,c = forward_activation(A_prev,W,b,activation = 'relu')
            >>> print("A with Relu = " + str(A))

            >>> A,c = forward_activation(A_prev,W,b,activation = 'softmax')
            >>> print("A with Softmax = " + str(A))
            
            Output:
                A with Relu = [[3.26295337 0.        ]]
                A with Softmax = [[1. 1.]]
    """
    
    if activation == 'relu':
        Z, sum_cache = forward_sum(A_prev,W,b)
        A, activation_cache = relu(Z)
        
    elif activation == 'softmax':
        Z, sum_cache = forward_sum(A_prev,W,b)
        A, activation_cache = softmax(Z)
    
    elif activation == "tanh":
#         Z, sum_cache = forward_sum(A_prev,W,b)
#         A, activation_cache = tanh(Z)
        pass
    
    cache = (sum_cache,activation_cache)
    
    assert(A.shape == Z.shape), "Dimention of A mismatched in forward_activation function"
    
    return A, cache

#-------------------------------------------------------------------------------------------------------------------
# dropout for individual layer
def forward_dropout(A,keep_probs):
     #implementing dropout
    D = np.random.rand(A.shape[0],A.shape[1])
    D = (D < keep_probs).astype(int)
    A = np.multiply(A,D)
    A = np.divide(A,keep_probs)
    
    dropout_mask = D
    
    assert (dropout_mask.shape == A.shape), "Dimention of dropout_mask mismatched in forward_dropout function"
    
    return A,dropout_mask

#-------------------------------------------------------------------------------------------------------------------
## forward prop for L layers
def forward_prop(X, parameters, keep_probs = [], regularizer = None):
    """
    
        Example:
            >>> np.random.seed(1)
            >>> X = np.random.randn(3,2)
            >>> W1 = np.random.randn(3,3)
            >>> b1 = np.random.randn(3,1)
            >>> W2 = np.random.randn(2,3)
            >>> b2 = np.random.randn(2,1)
            >>> parameters = {"W1": W1,
                              "b1": b1,
                              "W2": W2,
                              "b2": b2}
            >>> AL, caches, _ = forward_prop(X, parameters)
            >>> print("AL without dropout = " + str(AL))

            >>> AL, caches, _ = forward_prop(X, parameters,keep_probs = [0.9], regularizer = "dropout")
            >>> print("\nAL with dropout = " + str(AL))

            >>> print("\nLength of caches list = " + str(len(caches)))
            
            Output:
                AL without dropout = [[0.25442549 0.64096177]
                 [0.74557451 0.35903823]]

                AL with dropout = [[0.20251119 0.61487938]
                 [0.79748881 0.38512062]]

                Length of caches list = 2
    
    """
    caches = []
    A = X
    L = len(parameters) // 2
    num_class = parameters["W"+str(L)].shape[0]
    
    dropout_masks = []

    # len(keep_probs) == L-1: no dropouts in the Output layer, no dropout at all for prediction
    if regularizer == "dropout":
        assert(len(keep_probs) == L-1 ) 
    
    for l in range(1, L):
        A_prev = A 
        A, cache = forward_activation(A_prev,parameters['W' + str(l)],parameters['b' + str(l)], activation='relu')
        caches.append(cache)
        if regularizer == "dropout":
            A , dropout_mask = forward_dropout(A,keep_probs[l-1])
            dropout_masks.append(dropout_mask)
        else:
            pass

    AL, cache = forward_activation(A, parameters['W' + str(L)], parameters['b' + str(L)], activation='softmax')
    caches.append(cache)
    
    assert(AL.shape == (num_class,X.shape[1])), "Dimention of AL mismatched in forward_prop function"
    
    return AL,caches,dropout_masks
    
    
#====================================================================================================================
# compute Cross entropy cost
def softmax_cross_entropy_cost(AL, Y, caches, lambd = 0, regularizer = None, from_logits = False ):
    """
    
    
        Example:
            >>> AL = np.array([[4.21200131e-01, 1.55876995e-04],
                           [6.91917292e-02, 1.18118501e-05],
                           [5.09608140e-01, 9.99832311e-01]])
            >>> cost = softmax_cross_entropy_cost(AL, Y, caches)
            >>> print("Cost without l2 = " + str(cost))

            >>> cost = softmax_cross_entropy_cost(AL, Y, caches, lambd = 0.7, regularizer = 'l2')
            >>> print("Cost with l2 = " + str(cost))
            
            Output:
                Cost without l2 = 0.6742809046007259
                Cost with l2 = 8.875542970361
    """
    
    L = len(caches)
    m = Y.shape[1]
    
     #cost computation from logit
    #ref link : https://www.d2l.ai/chapter_linear-networks/softmax-regression-concise.html
    if from_logits == True:
        z = caches[-1][-1] #retriving the logit(activation cache) of the last layer from the caches
        
        z = z - np.max(z,axis = 0) #calculating negative z for avoiding numerical overflow in exp computation
        logit_log =  z - np.log(np.sum(np.exp(z),axis = 0)) #calculating the log of the softmax to feed into cost
        cost = -(1./m) * np.sum(np.sum(np.multiply(Y,logit_log), axis = 0,keepdims=True))
        
    else:
        cost = -(1./m) * np.sum(np.sum(np.multiply(Y,np.log(AL + 1e-8)), axis = 0,keepdims=True))# add very small number 1e-8 to avoid log(0)

    

    if regularizer == "l2":
        norm = 0
        for l in range(L):
            current_cache = caches[l]
            sum_cache, _ = current_cache
            _,W,_ = sum_cache
            norm += np.sum(np.square(W))

        L2_cost = (lambd/(2*m)) * norm 
        cost = cost + L2_cost
    else:
        pass
    
    cost = np.squeeze(cost)      # Making sure your cost's shape is not returned as ndarray
    
    assert(cost.shape == ()),"Dimention of cost mismatched in softmax_cross_entropy_cost function"
    
    return cost

#====================================================================================================================
# Back Propagation
#-------------------------------------------------------------------------------------------------------------------
## calculating backward gradient
def backward_grad(dZ, cache, lambd, regularizer):
    """
    
        Example:
            >>> np.random.seed(1)
            >>> dZ = np.random.randn(3,4)
            >>> A = np.random.randn(5,4)
            >>> W = np.random.randn(3,5)
            >>> b = np.random.randn(3,1)
            >>> cache = (A, W, b)
            
            >>> dA_prev, dW, db = backward_grad(dZ, cache, lambd=0, regularizer=None)
            >>> print("Without L2 Regularization")
            >>> print ("dA_prev = "+ str(dA_prev))
            >>> print ("dW = " + str(dW))
            >>> print ("db = " + str(db))
            
            >>> l2_dA_prev, l2_dW, l2_db = backward_grad(dZ, cache, lambd = 0.9, regularizer = 'l2')
            >>> print("\nWith L2 Regularization")
            >>> print ("dA_prev = "+ str(l2_dA_prev))
            >>> print ("dW = " + str(l2_dW))
            >>> print ("db = " + str(l2_db))
            
            Output:
                Without L2 Regularization
                dA_prev = [[-1.15171336  0.06718465 -0.3204696   2.09812712]
                           [ 0.60345879 -3.72508701  5.81700741 -3.84326836]
                           [-0.4319552  -1.30987417  1.72354705  0.05070578]
                           [-0.38981415  0.60811244 -1.25938424  1.47191593]
                           [-2.52214926  2.67882552 -0.67947465  1.48119548]]
                dW = [[ 0.07313866 -0.0976715  -0.87585828  0.73763362  0.00785716]
                      [ 0.85508818  0.37530413 -0.59912655  0.71278189 -0.58931808]
                      [ 0.97913304 -0.24376494 -0.08839671  0.55151192 -0.10290907]]
                db = [[-0.14713786]
                      [-0.11313155]
                      [-0.13209101]]

                With L2 Regularization
                dA_prev = [[-1.15171336  0.06718465 -0.3204696   2.09812712]
                           [ 0.60345879 -3.72508701  5.81700741 -3.84326836]
                           [-0.4319552  -1.30987417  1.72354705  0.05070578]
                           [-0.38981415  0.60811244 -1.25938424  1.47191593]
                           [-2.52214926  2.67882552 -0.67947465  1.48119548]]
                dW = [[-0.0814752  -0.28784277 -1.02688866  0.73478408 -0.24353767]
                      [ 0.90783172  0.74875962 -0.43216662  0.6696189  -0.78903459]
                      [ 0.81102242  0.13703735 -0.07696496  0.4081879  -0.05995309]]
                db = [[-0.14713786]
                      [-0.11313155]
                      [-0.13209101]]
    """
    
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    if regularizer == "l2":
        dW = (1/m) * np.dot(dZ,A_prev.T) + np.multiply(np.divide(lambd,m),W )
    else:
        dW = (1/m) * np.dot(dZ,A_prev.T)

    db = (1/m) * np.sum(dZ, axis = 1, keepdims=True )
    dA_prev = np.dot(W.T, dZ)

    
    assert (dW.shape == W.shape), "Dimention of dW mismatched in backward_grad function"
    assert (db.shape == b.shape), "Dimention of db mismatched in backward_grad function"
    assert (dA_prev.shape == A_prev.shape), "Dimention of dA_prev mismatched in backward_grad function"
    
    
    return dA_prev, dW, db

#-------------------------------------------------------------------------------------------------------------------
## calculating backward activation
def backward_activation(dA, cache, lambd ,regularizer, activation):
    """
        
        
        Example:
            >>> np.random.seed(2)
            >>> dA = np.random.randn(1,2)
            >>> A = np.random.randn(3,2)
            >>> W = np.random.randn(1,3)
            >>> b = np.random.randn(1,1)
            >>> Z = np.random.randn(1,2)
            >>> sum_cache = (A, W, b)
            >>> activation_cache = Z
            >>> cache = (sum_cache, activation_cache)
            
            >>> dA_prev, dW, db = backward_activation(dA, cache, lambd = 0 ,regularizer = None, activation = "relu")
            >>> print("With Relu")
            >>> print ("dA_prev = "+ str(dA_prev))
            >>> print ("dW = " + str(dW))
            >>> print ("db = " + str(db))
            
            >>> dA_prev, dW, db = backward_activation(dA, cache, lambd = 0 ,regularizer = None, activation = "softmax")
            >>> print("\nWith Softmax")
            >>> print ("dA_prev = "+ str(dA_prev))
            >>> print ("dW = " + str(dW))
            >>> print ("db = " + str(db))
            
            Output: 
                With Relu
                dA_prev = [[ 0.44090989 -0.        ]
                           [ 0.37883606 -0.        ]
                           [-0.2298228   0.        ]]
                dW = [[ 0.44513824  0.37371418 -0.10478989]]
                db = [[-0.20837892]]
            
                With Softmax
                dA_prev = [[ 0.44090989  0.05952761]
                           [ 0.37883606  0.05114697]
                           [-0.2298228  -0.03102857]]
                dW = [[ 0.39899183  0.3973954  -0.06975568]]
                db = [[-0.23651234]]
    """
    
    
    
    sum_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_grad(dA,activation_cache)
        dA_prev, dW, db = backward_grad(dZ, sum_cache, lambd, regularizer = regularizer)
        
    elif activation == "softmax":
        dZ = dA
        dA_prev, dW, db = backward_grad(dZ, sum_cache, lambd, regularizer = regularizer)
    
    elif activation == "tanh":
        pass
#         dZ = tanh_grad(dA,activation_cache)
#         dA_prev, dW, db = backward_grad(dZ, sum_cache, lambd, regularizer = regularizer)
    
    return dA_prev, dW, db
    
    
#-------------------------------------------------------------------------------------------------------------------
# implementing backward dropout
def backward_dropout(dA_prev_temp, D, keep_prob):
    dA_prev = np.multiply(dA_prev_temp,D)
    dA_prev = np.divide(dA_prev,keep_prob)
    
    return dA_prev

#-------------------------------------------------------------------------------------------------------------------
# back prop foL layers
def backward_prop(AL, Y, caches, dropout_masks = [], keep_probs = [], lambd = 0, regularizer = None):
    """
    
        Example:
            >>> np.random.seed(3)
            >>> AL = np.random.randn(1, 2)
            >>> Y = np.array([[1, 0]])

            >>> A1 = np.random.randn(4,2)
            >>> W1 = np.random.randn(3,4)
            >>> b1 = np.random.randn(3,1)
            >>> Z1 = np.random.randn(3,2)
            >>> cache_activation_1 = ((A1, W1, b1), Z1)

            >>> A2 = np.random.randn(3,2)
            >>> W2 = np.random.randn(1,3)
            >>> b2 = np.random.randn(1,1)
            >>> Z2 = np.random.randn(1,2)
            >>> cache_activation_2 = ((A2, W2, b2), Z2)

            >>> caches = (cache_activation_1, cache_activation_2)

            >>> grads = backward_prop(AL, Y, caches)
            >>> for key,value in grads.items():
            ...     print(str(key)+" : "+str(value))
            
            Output:
                dA1 : [[-0.80745758 -0.44693186]
                       [ 0.88640102  0.49062745]
                       [-0.10403132 -0.05758186]]
                dW2 : [[ 0.50767257 -0.42243102 -1.15550109]]
                db2 : [[0.61256916]]
                dA0 : [[ 0.          0.53064147]
                       [ 0.         -0.3319644 ]
                       [ 0.         -0.32565192]
                       [ 0.         -0.75222096]]
                dW1 : [[0.41642713 0.07927654 0.14011329 0.10664197]
                       [0.         0.         0.         0.        ]
                       [0.05365169 0.01021384 0.01805193 0.01373955]]
                db1 : [[-0.22346593]
                       [ 0.        ]
                       [-0.02879093]]
    """
    
    grads = {}
    L = len(caches) # the number of layers
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    dA = np.subtract(AL,Y)
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = backward_activation(dA, current_cache,lambd = lambd, regularizer = regularizer, activation = 'softmax')
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        
        if regularizer == "dropout":
            #implementing dropout
            D = dropout_masks[l]
            dA_prev_temp = backward_dropout(grads["dA" + str(l + 1)], D, keep_probs[l])
            dA_prev, dW_temp, db_temp = backward_activation(dA_prev_temp, current_cache, lambd = lambd, regularizer = regularizer, activation = 'relu')
        else:
            dA_prev, dW_temp, db_temp = backward_activation(grads["dA" + str(l + 1)], current_cache, lambd = lambd, regularizer = regularizer, activation = 'relu')
            
        
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

#====================================================================================================================
# Update Parameters
#-------------------------------------------------------------------------------------------------------------------
## Initializing Adam
def initialize_adam(parameters) :
   
    L = len(parameters) // 2 
    v = {}
    s = {}
    
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
    
    return v, s
#-------------------------------------------------------------------------------------------------------------------
## update Parameters
def update_parameters(parameters, grads, learning_rate, optimizer = "bgd", beta1 = 0, beta2 = 0,  epsilon = 0, v = {}, s = {}, t = 0):
    """
    
    
        Example:
            >>> np.random.seed(2)
            >>> W1 = np.random.randn(3,4)
            >>> b1 = np.random.randn(3,1)
            >>> W2 = np.random.randn(1,3)
            >>> b2 = np.random.randn(1,1)
            >>> parameters = {"W1": W1,
                          "b1": b1,
                          "W2": W2,
                          "b2": b2}
            >>> np.random.seed(3)
            >>> dW1 = np.random.randn(3,4)
            >>> db1 = np.random.randn(3,1)
            >>> dW2 = np.random.randn(1,3)
            >>> db2 = np.random.randn(1,1)
            >>> grads = {"dW1": dW1,
                     "db1": db1,
                     "dW2": dW2,
                     "db2": db2}

            >>> parameters,_,_ = update_parameters(parameters, grads, 0.1)

            >>> print ("W1 = "+ str(parameters["W1"]))
            >>> print ("b1 = "+ str(parameters["b1"]))
            >>> print ("W2 = "+ str(parameters["W2"]))
            >>> print ("b2 = "+ str(parameters["b2"]))
            
            Output:
                W1 = [[-0.59562069 -0.09991781 -2.14584584  1.82662008]
                      [-1.76569676 -0.80627147  0.51115557 -1.18258802]
                      [-1.0535704  -0.86128581  0.68284052  2.20374577]]
                b1 = [[-0.04659241]
                      [-1.28888275]
                      [ 0.53405496]]
                W2 = [[-0.55569196  0.0354055   1.32964895]]
                b2 = [[-0.84610769]]
    """
    L = len(parameters) // 2           
    v_corrected = {}                         
    s_corrected = {}                       
    
    for l in range(L):
        if optimizer == 'adam':
            # Moving average of the gradients.
            v["dW" + str(l+1)] = np.add(beta1 * v["dW" + str(l+1)], (1 - beta1) * grads["dW" + str(l+1)])
            v["db" + str(l+1)] = np.add(beta1 * v["db" + str(l+1)], (1 - beta1) * grads["db" + str(l+1)])

            # Compute bias-corrected first moment estimate.
            v_corrected["dW" + str(l+1)] = np.divide(v["dW" + str(l+1)], (1 - np.power(beta1,t)))
            v_corrected["db" + str(l+1)] = np.divide(v["db" + str(l+1)], (1 - np.power(beta1,t)))

            # Moving average of the squared gradients. 
            s["dW" + str(l+1)] = np.add(beta2 * s["dW" + str(l+1)], (1 - beta2) * np.square(grads["dW" + str(l+1)]))
            s["db" + str(l+1)] = np.add(beta2 * s["db" + str(l+1)], (1 - beta2) * np.square(grads["db" + str(l+1)]))

            # Compute bias-corrected second raw moment estimate. 
            s_corrected["dW" + str(l+1)] = np.divide(s["dW" + str(l+1)], (1 - np.power(beta2,t)))
            s_corrected["db" + str(l+1)] = np.divide(s["db" + str(l+1)], (1 - np.power(beta2,t)))

            # Update parameters. 
            parameters["W" + str(l+1)] = np.subtract(parameters["W" + str(l+1)],  learning_rate * np.divide(v_corrected["dW" + str(l+1)], np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon))
            parameters["b" + str(l+1)] = np.subtract(parameters["b" + str(l+1)],  learning_rate * np.divide(v_corrected["db" + str(l+1)], np.sqrt(s_corrected["db" + str(l+1)]) + epsilon))
        else:
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - (learning_rate * grads["dW" + str(l+1)])
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - (learning_rate * grads["db" + str(l+1)])
            
    return parameters, v, s

#====================================================================================================================
# Evaluating the model using acc and loss
def evaluate(X, Y, parameters):
    """
    
        Example:
            >>> np.random.seed(1)
            >>> X = np.random.randn(3,2)
            >>> Y = np.array([[1, 0, 0],[0,1,1]]).reshape(3,2)
            >>> W1 = np.random.randn(5,3)
            >>> b1 = np.random.randn(5,1)
            >>> W2 = np.random.randn(3,5)
            >>> b2 = np.random.randn(3,1)
            >>> parameters = {"W1": W1,
            ...               "b1": b1,
            ...               "W2": W2,
            ...               "b2": b2}
            >>> acc, loss = evaluate(X, Y, parameters)
            >>> print("acc = %f | cost = %f"%(acc,loss))
            
            Output:
                acc = 0.500000 | cost = 0.769464
    """
    
    m = Y.shape[1]
    
    # predicting output using fordward propogation 
    probas, caches, _ = forward_prop(X, parameters)
    #computing loss
    loss = softmax_cross_entropy_cost(probas, Y, caches, from_logits = True) 
    
    #deriving the predictrueted labels
    true_labels = np.argmax(Y,axis=0).reshape(1,m)
    #deriving the predicted labels
    predicted_labels = np.argmax(probas,axis=0).reshape(1,m)
    
    #identifing correctly predicted labels
    correct_prediction = np.equal(predicted_labels,true_labels)
    
    #computing accuracy
    num_correct_prediction = np.sum(correct_prediction)
    accuracy = (num_correct_prediction/m)
    
    return accuracy, loss
#====================================================================================================================
# learning rate scheduling
def learning_rate_scledule(alpha_prev, epoch, decay_rate = 1 ):
    alpha = (1/(1 + decay_rate * epoch)) * alpha_prev
    
    return alpha
#====================================================================================================================
# Final Model Training

def train(training_data, validation_data , layers_dim, hyperParams, initialization = "random", optimizer = 'bgd',regularizer = None, verbose = 3, patience = None, step_decay = None):
    # unpacking the hyperparameters
    learning_rate = hyperParams['learning_rate']
    num_epoch = hyperParams['num_epoch']
    b1 = hyperParams['beta1']
    b2 = hyperParams['beta2']
    ep = hyperParams['epsilon']
    lambd = hyperParams['lambda']
    keep_probs = hyperParams['keep_probs']
    
    # unpacking the data
    X_train, Y_train = training_data
    X_dev,Y_dev = validation_data

    # setting up necessary variables for early stopping
    if patience != None and patience !=0:
        # configuring path to save the intermediate best parameters
        path = "temp/" 
        if not os.path.exists(path):
            os.makedirs(path) 
        filename = "best_param_intermediate"

        early_stop_count = 0 # count variable for counting consucative the epochs without progress
        max_val_acc = 0 # for keeping track of maximum validation accuracy
    
    #initializing the training variables
    seed = 1
    m = Y_train.shape[1]
    train_accs = []  # for keeping track of training accuracy
    val_accs = []     # for keeping track of Validation accuracy
    train_losses = []  # for keeping track of training loss
    val_losses = []     # for keeping track of Validation loss
    
    #selecting the minibatch size for each optimizer
    if optimizer == 'sgd':
        mini_batch_size = 1
    elif optimizer == 'bgd':
        mini_batch_size = m
    elif optimizer == 'mgd' or optimizer == 'adam':
        mini_batch_size = hyperParams['mini_batch_size']
    else:
        raise ValueError("Optimizer value out of scope")
        
    #initializing the model parameters
    parameters = init_parameters(layers_dim, initialization)
    
    #initializing adam parameters, used only when optimizer = 'adam'
    t = 0
    v,s = initialize_adam(parameters)
    
    train_tic = time.time() # for calculating entire training time
    print("Training The Model...")
    
    #Gradient Descent begins
    for i in range(num_epoch):
        seed += 1
        time_trained = 0 # for computing training time of each epoch
        batch_times = [] # for accumulating the training time of each batch
        accs = [] # for tracking batch training accuracy
        losses = [] # for tracking batch training loss
        
        #learning rate scheduling
        if step_decay!= None and step_decay!= 0:
            if i%step_decay == 0:
                decay_rate = learning_rate / ((i+1)/num_epoch)
                learning_rate = learning_rate_scledule(learning_rate, i, decay_rate)
                if learning_rate <= 0.0001: learning_rate = 0.0001 
        
           
        if verbose > 0:
            if step_decay!= None and step_decay!= 0:
                print("\nEpoch %d/%d: learning rate - %.6f"%(i+1,num_epoch,learning_rate))
            else:
                print("\nEpoch %d/%d"%(i+1,num_epoch))
                
        #generating minimatches
        minibatches = rand_mini_batches(X_train, Y_train, mini_batch_size, seed)
        total_minibatches = len(minibatches)
        
        for ind, minibatch in enumerate(minibatches):
            batch_tic = time.time() # for calculating time of an epoch cycle
            
            #retriving minibatch of X and Y from training set
            (minibatch_X, minibatch_Y) = minibatch
            
            #forward Propagation
            AL, caches, dropout_masks = forward_prop(minibatch_X, parameters, keep_probs = keep_probs, regularizer = regularizer)
            
            #Computing cross entropy cost
            cross_entropy_cost = softmax_cross_entropy_cost(AL, minibatch_Y, caches, lambd = lambd, regularizer = regularizer, from_logits = True) #accumulating the batch costs
            
            #Backward Propagation
            grads = backward_prop(AL, minibatch_Y, caches, dropout_masks = dropout_masks, keep_probs = keep_probs, lambd = lambd, regularizer = regularizer)
                
            #Updating parameters
            t += 1
            parameters, v, s = update_parameters(parameters, grads, learning_rate, optimizer = optimizer, beta1 = b1, beta2 = b2,  epsilon = ep, v = v, s = s, t = t)
            
            # Calculating training time for each batch 
            batch_times.append(time.time() - batch_tic)
            time_trained = np.sum(batch_times)
            
            #calculating training progress
            per = ((ind+1) / total_minibatches) * 100
            inc = int(per // 10) * 2
            
            #calculating accuracy and loss of the training batch
            acc,loss = evaluate(minibatch_X, minibatch_Y, parameters)
            accs.append(acc)
            losses.append(loss)
            
            
            #Verbosity 0: Silent mode
            #Verbosity 1: Epoch mode
            #Verbosity 2: Progress bar mode
            #Verbosity 3 or greater: Metric mode
                
            if verbose == 2:
                print ("%d/%d [%s>%s %.0f%%] - %.2fs"%(ind+1, total_minibatches, '=' * inc,'.'*(20-inc), per, time_trained),end='\r')
            elif verbose > 2:
                print ("%d/%d [%s>%s %.0f%%] - %.2fs | loss: %.4f | acc: %.4f"%(ind+1, total_minibatches, '=' * inc,'.'*(20-inc), per, time_trained, np.mean(losses), np.mean(accs)),end='\r')
            
        #----------------------------------------------batch ends-------------------------------------------
        
        #accumulating the acc and loss of the last iteration of each epoch
        train_accs.append(np.mean(accs))
        train_losses.append(np.mean(losses))
                
        #evaluating the model using validation accuracy and loss
        val_acc, val_loss= evaluate(X_dev, Y_dev, parameters)  
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        
        time_per_batch = int(np.mean(batch_times)*1000)

        if verbose == 2:
            print ("%d/%d [%s 100%%] - %.2fs %dms/step"%(total_minibatches, total_minibatches, '=' * 20, time_trained, time_per_batch ),end='\r')
        elif verbose > 2:
            print ("%d/%d [%s 100%%] - %.2fs %dms/step | loss: %.4f | acc: %.4f | val_loss: %.4f | val_acc: %.4f"%(total_minibatches, total_minibatches, '=' * 20, time_trained, time_per_batch, np.mean(losses), np.mean(accs), val_loss, val_acc),end='\r')
                
        #early stopping implementation
        if patience != None and patience !=0:
            #getting the best val accuracy
            if val_acc >= max_val_acc:
                max_val_acc = val_acc
                print("\nImprovement in validation accuracy found. Saving the corresponding parameters...")
                save_model(path+filename, parameters)
                early_stop_count = 0
            else:
                early_stop_count += 1
            
            if early_stop_count == patience:
                print("\n\nSince the Val Acc didn't increase for last %d epochs, Training is halted returning the best parameters obtained."%patience)
                break;
                
    #-------------------------------------------Gradient Descent ends-----------------------------------------------
    
    hrs, mins, secs , ms = convert_time((time.time() - train_tic)*1000)
    print("\n\nTotal Training Time = %dhr %dmins %dsecs %.2fms"%(hrs, mins, secs, ms))
    
    #loading the best parameters
    if patience != None and patience !=0:
        parameters = load_model(path+filename)
        os.remove(path + filename) #removing temporary file
        
    history = {"parameters":parameters,
               "accuracy": train_accs,
               "loss":train_losses ,
               "val_accuracy":val_accs,
               "val_loss":val_losses
            }
    return history

#====================================================================================================================
#making Prediction
# Making Predictions
def predict(X, parameters, second_guess = False):
    """
    
    """
    prediction = {}
    
    # Computing the Output predictions. 
    # no keep_probs : no dropout during prediction 
    probas, caches, _ = forward_prop(X, parameters)
    
    #getting the number of examples
    m = probas.shape[1]

    #deriving the predicted labels with their probabilities
    predicted_labels = np.argmax(probas,axis=0).reshape(1,m)
    predicted_prob = np.max(probas,axis = 0).reshape(1,m)
    
    #Computing the second guess
    if second_guess == True:
        second_max = np.array(probas, copy=True)
        second_max[predicted_labels,np.arange(m)] = 0 #zeroing out the first max prediction
        sec_predicted_labels = np.argmax(second_max,axis=0).reshape(1,m) #selecting the second max predicted label
        sec_predicted_prob = np.max(second_max,axis = 0).reshape(1,m) #selecting the second max prediction

        prediction["Second Prediction"] = [sec_predicted_labels, sec_predicted_prob]      

    prediction["First Prediction"] = [predicted_labels, predicted_prob]
    

    return prediction