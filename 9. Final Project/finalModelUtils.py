#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:41:53 2020

@author: befrenz
"""
import numpy as np
import pickle #for saving model
import matplotlib.pyplot as plt
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

#====================================================================================================================
# Visualizing the Training Result

def visualize_training_results(train_accs, val_accs, train_loss, val_loss):
    
    #creating subplots
    fig, axes = plt.subplots(nrows=2, ncols = 1,figsize=(10,10))
    fig.subplots_adjust(wspace=.2, hspace = .5)
    
    axes[0].plot(np.squeeze(train_loss), label = 'Training Loss', color = 'blue')
    axes[0].plot(np.squeeze(val_loss), label = 'Validation Loss', color = 'red')
    axes[0].legend(loc='center right')
    
    axes[0].set_title("Training and Validation Loss " , fontsize = 16, pad = 10)
    axes[0].set_xlabel("No. of Epochs", fontsize = 12)
    axes[0].set_ylabel("Loss", fontsize = 12)
    axes[0].set_ylim(top = 1, bottom = -0.5)  
    axes[0].grid(True)
    
    axes[1].plot(np.squeeze(train_accs), label = 'Training Accuracy', color = 'blue')
    axes[1].plot(np.squeeze(val_accs), label = 'Validation Accuracy', color = 'red')
    axes[1].legend(loc='center right')
    axes[1].set_title("Accuracy " , fontsize = 16, pad = 10)
    axes[1].set_xlabel("No. of Epochs", fontsize = 12)
    axes[1].set_ylabel("Accuracy", fontsize = 12)
    axes[1].set_ylim(bottom = 0.85)  
    axes[0].grid(True)

    plt.show()

#====================================================================================================================
#time Conversion
def convert_time(millisec):
    
    hours = millisec // 3.6e+6
    mins = (millisec % 3.6e+6) // 60000
    secs = ((millisec % 3.6e+6) % 60000) // 1000
    millisec = ((millisec % 3.6e+6) % 60000) % 1000
    
    return (hours,mins,secs, millisec)

#====================================================================================================================
#creating Confusion Matrix
def confusion_matrix(y_orig,prediction):
    first_predict = prediction["First Prediction"]

    y_predicted = first_predict[0]
    y_predicted = y_predicted.T
    
    m = y_orig.shape[0]
    k = len(np.unique(y_orig)) # or simply take k =10
    
    cm = np.zeros((k,k))

    for i in range(m):
        cm[y_orig[i],y_predicted[i]] += 1
   
    return cm.astype(int)

#-------------------------------------------------------------------------------------------------------------------
## Plotting Confusion Matrix

def plot_confusion_matrix(cm, dataset):
    fig, ax = plt.subplots(figsize=(10,10))
    im = plt.imshow(cm,cmap="GnBu") #RdYlGn, PiYG, Accent,Blues,viridis, YlGnBu


    fig.colorbar(im,ax=ax,fraction=0.045)
    # ax.set_aspect('auto')
    
    if(dataset == "training"):
        visual_title = "Confusion Matrix for Training Set "
    elif(dataset == "dev"):
        visual_title = "Confusion Matrix for Dev Set "
    elif(dataset == "test"):
        visual_title = "Confusion Matrix for Test Set "
    else:
        raise ValueError("Dataset set must be training or dev or test set")
    
    ax.set_title(visual_title,fontsize=24,pad = 20)
    ax.set_xticks(range(0,10))
    ax.set_yticks(range(0,10))
    ax.set_xlabel("Predicted", fontsize = 20)
    ax.set_ylabel("Expexted", fontsize = 20)

    ax.set_xticklabels([0,1,2,3,4,5,6,7,8,9],fontsize=16)
    ax.set_yticklabels([0,1,2,3,4,5,6,7,8,9],fontsize=16)

    #setting horizontal axes labeling to top.
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')


    # Turn off all the ticks
    ax.tick_params(top=False,left=False)


    thres = cm.max()//2

    for i in range(10):
        for j in range(10):
            per = cm[i,j]/cm.sum() * 100
            ax.text(j, i, "%d\n%.2f%%"%(cm[i, j], per),
                           ha="center", va="center", color="w" if cm[i,j] > thres else "black")


    fig.tight_layout()
    plt.show()

#-------------------------------------------------------------------------------------------------------------------
## Calculating precision, Recall and F1-Score
def precision(label, cm):
    col = cm[:, label]
    prec = cm[label, label] / col.sum()
    return prec
    
def recall(label, cm):
    row = cm[label, :]
    rec = cm[label, label] / row.sum()
    return rec
def f1_score(prec,rec):
    f1 = (2 * prec * rec) / (prec + rec)
    return f1

#-------------------------------------------------------------------------------------------------------------------
##Calculating macro precision, recall and f1-score
def macro_precision_average(prec):
    count = len(prec)    
    prec_mac_avg = np.sum(prec) / count
    return prec_mac_avg

def macro_recall_average(rec):
    count = len(rec)
    rec_mac_avg = np.sum(rec) / count
    return rec_mac_avg

def macro_f1_score(f1):
    count = len(f1)
    f1_mac_avg = np.sum(f1) / count
    return f1_mac_avg

#-------------------------------------------------------------------------------------------------------------------
##calculating the accuracy
def accuracy(cm):
    diagonal_sum = cm.trace()
    sum_of_all_elements = cm.sum()
    acc = diagonal_sum / sum_of_all_elements 
    return acc

#-------------------------------------------------------------------------------------------------------------------
## Calculating model metrices
def model_metrics(cm):
    prec = []
    rec = []
    f1 = []
    metrics = {}
    macro_metrics = {}
    for label in range(10):
        prec.append(precision(label, cm))
        rec.append(recall(label, cm))
        f1.append(f1_score(prec[label], rec[label]))
    
    avg_precision = macro_precision_average(prec)
    avg_recall = macro_recall_average(rec)
    avg_f1 = macro_f1_score(f1)
    acc = accuracy(cm)
    
    metrics = {"Precision":prec,
               "Recall":rec,
               "F1-Score":f1}
    macro_metrics = {"Precision":avg_precision,
                     "Recall":avg_recall,
                     "F1-Score":avg_f1}
    
    return metrics, macro_metrics, acc

#-------------------------------------------------------------------------------------------------------------------

##displaying the model summary
def metric_summary(metrics, macro_metrics, accuracy):
    print("+===============+===============+===============+===============+")
    print("| Label \t| Precision \t| Recall \t| F1 Score \t|")
    print("+===============+===============+===============+===============+")
    prec = metrics["Precision"]
    rec = metrics["Recall"]
    f1 = metrics["F1-Score"]
    
    for label in range(len(prec)):
        print("| %d \t\t|  %.5f \t|  %.5f \t|  %.5f \t|"%(label, prec[label], rec[label], f1[label]))

    print("+===============+===============+===============+===============+") 
    
    avg_precision = macro_metrics["Precision"]
    avg_recall = macro_metrics["Recall"]
    avg_f1 = macro_metrics["F1-Score"]
    acc = accuracy
    
    print("| Macro Avg \t|  %.5f \t|  %.5f \t|  %.5f \t|"%( avg_precision, avg_recall, avg_f1))
    print("+===============+===============+===============+===============+") 
    
    print("\n Accuracy \t\t  %.5f"%(acc))
    
    return 

#====================================================================================================================
#Visualizing Prediction
def visualize_prediction(x_orig, y_orig, prediction, dataset):
    if(dataset == "training"):
        visual_title = "Sample Training Data Set"
        rng = range(30,40)
    elif(dataset == "dev"):
        visual_title = "Sample Dev Data Set"
        rng = range(110,120)
    elif(dataset == "test"):
        visual_title = "Sample Test Data Set"
        rng = range(110,120)        
    else:
        raise ValueError("Dataset set must be training or dev or test set")
    fig, axes = plt.subplots(nrows=2, ncols=4,figsize=(16,8))
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(visual_title)

    first_lbl, first_prob = prediction["First Prediction"]
    sec_lbl, sec_prob = prediction["Second Prediction"]

    for ax,i in zip(axes.flatten(),rng):
        ax.imshow(x_orig[i].squeeze(),interpolation='nearest', cmap='Greys')
        ax.set(title = "True: "+ str(y_orig[0,i]))
        ax.set(xlabel= "Prediction: %d | With Prob: %.4f \n2nd Guess: %d | With Prob: %.4f"%(first_lbl[0,i], first_prob[0,i], sec_lbl[0,i], sec_prob[0,i]))

#====================================================================================================================
# Visualizing Mislabeled Data
def visualize_mislabelled_images(x_orig,y_orig,prediction,dataset):
    
    first_lbl, first_prob = prediction["First Prediction"]
    sec_lbl, sec_prob = prediction["Second Prediction"]
    
    true_prediction = np.equal(first_lbl,y_orig)
    mislabelled_indices = np.asarray(np.where(true_prediction == False))
    print("Total Mislabelled Images: "+str(len(mislabelled_indices[0])))
    
    if(dataset == "training"):
        visual_title = "Sample Mislabelled Training Images"
    elif(dataset == "dev"):
        visual_title = "Sample Mislabelled Dev Images"
    elif(dataset == "test"):
        visual_title = "Sample Mislabelled Test Images"
    else:
        raise ValueError("Dataset set must be training or dev or test set")
    
    fig, axes = plt.subplots(nrows=2, ncols=5,figsize=(16,8))
    fig.subplots_adjust(hspace=1)
    fig.suptitle(visual_title)

    
    
    for ax,i in zip(axes.flatten(),mislabelled_indices[1]):
        ax.imshow(x_orig[i].squeeze(),interpolation='nearest')
        ax.set(title = "True: "+ str(y_orig[0,i]))
        ax.set(xlabel= "Prediction: %d | With Prob: %.4f \n2nd Guess: %d | With Prob: %.4f"%(first_lbl[0,i], first_prob[0,i], sec_lbl[0,i], sec_prob[0,i]))

#====================================================================================================================
#saving Model
def save_model(file_name, parameters):
    
        with open(file_name ,'wb') as output_file:
            pickle.dump(parameters,output_file)

#====================================================================================================================
#Loading Model
def load_model(file_name):
    try: 
        with open(file_name ,'rb') as input_file:
            parameters = pickle.load(input_file)
        
        return parameters
    
    except(OSError, IOError) as e:
        print(e)

#====================================================================================================================
