#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:04:49 2020

@author: befrenz
"""
# Python Standard Libraries for importing data from binary file
import os.path #for accessing the file path
import struct  #for unpacking the binary data

#core packages
import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------------------------------
#data retriving
def retrive_data(dataset="training-set"):
    """
        Retrive MNIST dataset from  the binary file into numpy arrays        
        
        Dataset Obtained From:
            - link -- http://yann.lecun.com/exdb/mnist/
            
        Dataset retrival code adapted from(but modified to our need making data retrival 6-8 times faster):
            - link -- https://www.cs.virginia.edu/~connelly/class/2015/large_scale/proj2/mnist_python
            
        Argument:
            - **dataset** -- type of dataset to be loaded. may be either 'training' or 'test'
        Returns:
            - **images** -- 3D array consisting of no. of examples, rows, columns of images 
            - **labels** -- array  containing labels for each images
    """
    path = "dataset/"
    size = 60000
    
    #setting file path based on the dataset
    if dataset == "training-set":
        img_file_path = os.path.join(path, 'train-images-idx3-ubyte')
        lbl_file_path = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "test-set":
        img_file_path = os.path.join(path, 't10k-images-idx3-ubyte')
        lbl_file_path = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("Dataset must be 'training-set' or 'test-set'")
    
    #retriving the data
    with open(lbl_file_path, 'rb') as flbl:
        _, size = struct.unpack(">II", flbl.read(8))
        labels = np.frombuffer(flbl.read(), dtype=np.int8).reshape(size,1)

    with open(img_file_path, 'rb') as fimg:
        _, _, rows, cols = struct.unpack(">IIII", fimg.read(16))
        images = np.frombuffer(fimg.read(),dtype=np.uint8).reshape(size, rows, cols)
        
    assert(images.shape == (size, rows, cols))
    assert(labels.shape == (size,1))
    
    return images, labels


#-----------------------------------------------------------------------------------------------------------------
#splitting the test set into dev and test set
def dev_test_split(test_x,test_y):
    """
        Randomly splits the test set to dev and test set
        
        Arguments:
            - **test_x** -- test set images of size (10000,28,28) 
            - **test_y** -- test set labels of size (10000,1)
        
        Returns:
            - **dev_x**  -- dev set images of size (n,28,28) 
            - **dev_y**  -- dev set labels of size (n,1) 
            - **test_x** -- test set images of size (n,28,28) 
            - **test_y** -- test set labels of size (n,1)
    """
    m = test_y.shape[0]
    n = m // 2

    #suffling the test dataset
    randCol = np.random.permutation(m)
    suffled_x = test_x[randCol,:,:]
    suffled_y = test_y[randCol,:]
    
    #splitting the test set into dev and test set , 50% each
    dev_x = suffled_x[0:n,:,:]
    dev_y = suffled_y[0:n,:]
    
    test_x = suffled_x[n:m,:,:]
    test_y = suffled_y[n:m,:]
    
    assert(dev_x.shape == (n,28,28))
    assert(dev_y.shape == (n,1))
    assert(test_x.shape == (n,28,28))
    assert(test_y.shape == (n,1))
    
    return dev_x,dev_y,test_x,test_y

#-----------------------------------------------------------------------------------------------------------------
#retriving a small sample of the original dataset for model development and experimentation
def sample_origDataset(x,y, sample_size):
    """
        Returns a sample dataset from the fully processed dataset
       
        Arguments:
            - **x** -- original input data
            - **y** -- original output labels
            - **sample_size** -- sample volume in percentage
        Returns:
            - **x_sample** -- input sample  from original dataset of size ( dataVol% of x)
            - **y_sample** -- output sample  from original dataset of size (datavol% of y)
    """
    m = y.shape[0]
    sample_m = int(np.multiply(m,np.divide(sample_size,100))) #int(m*(dataVol/100)) 
    
    #suffling the original dataset
    randCol = np.random.permutation(m)
    x_suffled = x[randCol,:,:]
    y_suffled = y[randCol,:]
    
    #taking samples of sample_size
    x_sample = x_suffled[0:sample_m,:,:]
    y_sample = y_suffled[0:sample_m,:]

    assert(x_sample.shape == (sample_m,28,28))
    assert(y_sample.shape == (sample_m,1))

    return x_sample, y_sample
#-----------------------------------------------------------------------------------------------------------------

#loading the entire dataset
def load_dataset():
    """
        Retrive the dataset from file into training, dev and test sets.
        
        Returns: 
            - **train_x_orig** --  training set images consisting of no. of examples, rows, columns of images, size(60000,28,28) 
            - **train_y_orig** --  training set output consisting of image labels, size(60000,1) 
            - **dev_x_orig**  -- dev set images of size (5000,28,28) 
            - **dev_y_orig**  -- dev set labels of size (5000,1) 
            - **test_x_orig** -- test set images of size (5000,28,28) 
            - **test_y_orig** -- test set labels of size (5000,1) 
        
    """
    #retriving data
    train_x_orig, train_y_orig = retrive_data(dataset="training-set")
    test_x_temp, test_y_temp = retrive_data(dataset="test-set")
    
    #Spliting the test set into dev and test set
    dev_x_orig,dev_y_orig,test_x_orig,test_y_orig = dev_test_split(test_x_temp, test_y_temp)
    
    return train_x_orig, train_y_orig, dev_x_orig,dev_y_orig,test_x_orig,test_y_orig

#-----------------------------------------------------------------------------------------------------------------

#loading sample dataset
def load_sample_dataset(sample_size):
    """
        Retrive the sample dataset from file into training, dev and test sets.
        
        Argument:
            - **sample_size** -- sample volume in percentage

        Returns: 
        - **train_x_sample** -- input sample from training set of size ( sample_size% of train_x_orig)
        - **train_y_sample** -- output sample from training set of size (sample_size% of train_y_orig)
        - **dev_x_sample** -- input sample from dev set of size ( sample_size% of dev_x_orig)
        - **dev_y_sample** -- output sample from dev set of size (sample_size% of dev_y_orig)
        - **test_x_sample** -- input sample from test set of size ( sample_size% of test_x_orig)
        - **test_y_sample** -- output sample from test set of size (sample_size% of test_y_orig)        
    """
    #loading the original dataset
    train_x_orig, train_y_orig, dev_x_orig,dev_y_orig,test_x_orig,test_y_orig = load_dataset()

    
    #sampling from the dataset
    train_x_sample, train_y_sample = sample_origDataset(train_x_orig, train_y_orig, sample_size)
    dev_x_sample,dev_y_sample = sample_origDataset(dev_x_orig, dev_y_orig, sample_size)
    test_x_sample,test_y_sample = sample_origDataset(test_x_orig, test_y_orig, sample_size)
    

    return train_x_sample, train_y_sample, dev_x_sample, dev_y_sample, test_x_sample, test_y_sample

#-----------------------------------------------------------------------------------------------------------------
#dataset visualization using charts
def visual_charts(train_y_orig, dev_y_orig, test_y_orig):
    """
        Plots bar graph showing the number of examples in each class

        Arguments:
            trainy_orig - labels of training set
            dev_y_orig - labels of dev set
            test_y_orig - labels of test set
    """
    datasets = {"Training Set":train_y_orig,"Dev Set": dev_y_orig,"Test Set": test_y_orig}
    
    #setting the plot style
#     plt.style.use('seaborn')
    
    #creating subplots
    fig, axes = plt.subplots(nrows=3, ncols=1,figsize=(10,15))
    fig.subplots_adjust(hspace=.2)
    i = 0
    
    #plotting the bar graph for each dataset labels
    for dataset,datalabel in datasets.items():
        unique, counts = np.unique(datalabel, return_counts=True)    
        axes[i].bar(unique, counts)
        max_value = np.max(counts)
        axes[i].set(xticks = unique, ylim = (0,max_value + max_value // 10))
        axes[i].set_title("Number of Examples in " + dataset , fontsize = 16)
        axes[i].set_xlabel("Classes", fontsize = 12)
        axes[i].set_ylabel("Number of Examples", fontsize = 12)
        i += 1

    plt.show()
#-----------------------------------------------------------------------------------------------------------------
#visualizing the loaded dataset
def visualize_dataset(x_orig, y_orig, dataset = "training"):
    """
        Plots 10 sample images from the dataset with labels
        
        Arguments:
            x_orig - 3D array representation of input images
            y_orig - array of labels
            dataset - type of dataset, can be training, dev or test
        
    """
    #recovering matplotlib defaults
#     plt.rcParams.update(plt.rcParamsDefault) 
    
    #checking dataset type
    if(dataset == "training"):
        visual_title = "Training Data Set"
        rng = range(1040,1050)
    elif(dataset == "dev"):
        visual_title = "Dev Data Set"
        rng = range(100,110)
    elif(dataset == "test"):
        visual_title = "Test Data Set"
        rng = range(540,550)        
    else:
        raise ValueError("Dataset set must be training or dev or test set")
     
    #creating subplots
    fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(16,8))
    fig.subplots_adjust(hspace=.1)
    fig.suptitle(visual_title)
    
    #plotting the sample images along with their labels
    for ax,i in zip(axes.flatten(),rng):
        ax.imshow(x_orig[i].squeeze(),interpolation='nearest')
        ax.set(title = "Label: "+ str(y_orig[i,0]))
#-----------------------------------------------------------------------------------------------------------------
#flatten the input images      
def flatten_input(train_x_orig,dev_x_orig,test_x_orig):
    """
        Flattens the 3D numpy array of the input images
        
        Arguement:
            - **train_x_orig** --  training set images of size (60000,28,28) 
            - **dev_x_orig**   -- dev set images of size (5000,28,28) 
            - **test_x_orig**  -- test set images of size (5000,28,28) 

        Returns:
            - **train_x_flatten** -- flattened training set input data of size (784,60000) 
            - **dev_flatten**     -- flattened training set dev data of size (784,5000) 
            - **test_x_flatten**  -- flattened test set input data of size (784,5000) 
            
    """
    m = train_x_orig.shape[0] #number of examples in training set
    n = dev_x_orig.shape[0] # number of examples in dev and test set
    
    
    #flattening the image--The "-1" makes reshape flatten the remaining dimensions
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   
    dev_x_flatten = dev_x_orig.reshape(dev_x_orig.shape[0], -1).T    
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
   
    
    assert(train_x_flatten.shape == (784,m) )
    assert(dev_x_flatten.shape == (784,n) )
    assert(test_x_flatten.shape == (784,n) )
    
    return train_x_flatten, dev_x_flatten, test_x_flatten

#-----------------------------------------------------------------------------------------------------------------
#normalize the input images
def normalize_input(train_x_flatten,dev_x_flatten,test_x_flatten ):
    """
        Normalizes the pixel values of the flattened images to the range 0-1
        
        Arguement:
            - **train_x_flatten** -- flattened training set input data of size (784,60000) 
            - **dev_flatten**     -- flattened training set dev data of size (784,5000) 
            - **test_x_flatten**  -- flattened test set input data of size (784,5000) 
        Returns:
            - **train_x_norm** -- normalized training set input data 
            - **dev_norm**     -- normalized training set dev data 
            - **test_x_norm**  -- normalized test set input data
    """
    m = train_x_flatten.shape[1]
    n = dev_x_flatten.shape[1]
    # Normalizing the data into the range between 0 and 1.
    train_x_norm = np.divide(train_x_flatten,255.)
    dev_x_norm = np.divide(dev_x_flatten,255.)
    test_x_norm = np.divide(test_x_flatten,255.)
    
    assert(train_x_norm.shape == (784,m) )
    assert(dev_x_norm.shape == (784,n) )
    assert(test_x_norm.shape == (784,n) )
    
    return train_x_norm, dev_x_norm, test_x_norm

#-----------------------------------------------------------------------------------------------------------------
# incode the output labels into one-hot representation
def one_hot_encoding(y_orig,num_classes = 10):
    """
        Transform the output labels into the one-hot encoding representation
        
        Arguments:
            - **y_orig** -- raw labels loaded directly from the binary file
            - **num_classes** -- number of the classes based on which the transformation is to be made
        Returns:
            - **y_encoded** -- encoded ndarray of the labels with data elements of int type
    """
    #encoding the labels
    y_encoded = np.eye(num_classes)[y_orig.reshape(-1)].T


    assert(y_encoded.shape == (num_classes, y_orig.shape[1]))
    return y_encoded

#-----------------------------------------------------------------------------------------------------------------
#preparation of the entire dataset before training
def prep_dataset(train_x_orig, train_y_orig, dev_x_orig, dev_y_orig, test_x_orig, test_y_orig):
    """
        Flatten and Normalize the input images and encode the output labels
        
        Arguments:
            - **train_x_orig** --  training set images of size (60000,28,28)
            - **train_y_orig** --  training set labels of size (60000,1)
            - **dev_x_orig**   -- dev set images of size (5000,28,28)
            - **dev_y_orig**   -- dev set labels of size (5000,1)
            - **test_x_orig**  -- test set images of size (5000,28,28)
            - **test_y_orig**  -- test set labels of size (5000,1)
        Returns:
            - **train_x_norm** -- flattened and normalized training set input data
            - **dev_norm**     -- flattened and normalized training set dev data
            - **test_x_norm**  -- flattened and normalized test set input data
            - **train_y_encoded** -- encoded label of training set
            - **dev_y_encoded**   -- encoded label of dev set
            - **test_y_encoded**  -- encoded label of test set
    """
    #flatten the input images
    train_x_flatten,dev_x_flatten,test_x_flatten = flatten_input(train_x_orig,dev_x_orig,test_x_orig)
    
    #normalize the input images
    train_x_norm, dev_x_norm, test_x_norm = normalize_input(train_x_flatten,dev_x_flatten,test_x_flatten)
    
    #encode the output labels
    train_y_encoded = one_hot_encoding(train_y_orig.T)
    dev_y_encoded = one_hot_encoding(dev_y_orig.T)
    test_y_encoded = one_hot_encoding(test_y_orig.T)
    
    return train_x_norm,train_y_encoded, dev_x_norm,dev_y_encoded, test_x_norm, test_y_encoded

#-----------------------------------------------------------------------------------------------------------------