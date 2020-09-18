#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:26:57 2020

@author: befrenz
"""
# ===================================================( Loading Dependencies )===================================================
# Python Standard Libraries
import pickle  # for saving model

# core packages
import numpy as np
import matplotlib.pyplot as plt

# loading custom model
from dataset import label_description


# Doc String is based on Google Style Python Docstrings: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html
# Doc Rendering:sphinxcontrib.napoleon 
# ===================================================( Activation Functions )===================================================
def relu(Z):
    """Compute the ReLU activation of Z.
        
        Arguments:
            Z (numpy.ndarray): Input Sum to a hidden unit, Z = W * X + b.
        
        Returns:
            tuple: Following values
            - **A** (numpy.ndarray): Activation obtained by applying ReLU function to the sum. Size same as that of Z.
            - **cache** (numpy.ndarray): Value stored for use during backward propagation.
        
        Example:
            >>> np.random.seed(1)
            >>> Z = np.random.randn(1,6)
            >>> A,cache = relu(Z)
            >>> print(A)
            
            Output: 
                [[1.62434536 0.     0.      0.      0.86540763 0.    ]]
    """
    A = np.maximum(0.0, Z)

    cache = Z  # storing Z for later use during back propagation

    assert (A.shape == Z.shape)
    return A, cache


# ------------------------------------------------------------------------------------------------------------------------------
def relu_grad(dA, cache):
    """Compute the backward ReLU activation of dA.
        
        Arguments:
            dA (numpy.ndarray): Gradient of activation of the previous layer.
            cache (numpy.ndarray): Value of Z stored during forward prop.
            
        Returns:
            numpy.ndarray: - **dZ**: array of gradient/derivative of the dA, Same size of dA.
            
        Example:
            >>> np.random.seed(1)
            >>> dA = np.random.randn(1,6)
            >>> cache = np.random.randn(1,6)
            >>> dZ = relu_grad(dA,cache)
            >>> print(dZ)
            
            Output:
                [[ 1.62434536  0.         -0.52817175  0.          0.86540763  0.        ]]
    """
    Z = cache
    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = 0  # implementing integrated form of (gradiant of ReLU function * gradient of the loss function)

    assert (dZ.shape == Z.shape)
    return dZ


# ------------------------------------------------------------------------------------------------------------------------------
def softmax(Z):
    """Compute the softmax activtion of Z.
        
        Argument:
            Z (numpy.ndarray): Input Sum to a hidden unit, Z = W * X + b.
        
        Returns:
            tuple: Following Values
            - **A** (numpy.ndarray): Activation obtained by applying softmax function to the sum. Size same as that of Z.
            - **cache** (numpy.ndarray): Value stored for use during backward propagation.
            
        Example:
            >>> np.random.seed(2)
            >>> Z= np.random.rand(7,1)
            >>> A,cache = softmax(Z)
            >>> print(A)
            
            Output:
                [[0.15477477]
                 [0.10270926]
                 [0.17340649]
                 [0.15467071]
                 [0.15237489]
                 [0.13925557]
                 [0.12280831]]
    """
    shift = Z - np.max(Z, axis = 0)# Avoiding underflow or overflow errors due to floating point instability in softmax
    t = np.exp(shift)
    A = np.divide(t, np.sum(t, axis=0))

    cache = Z

    assert (A.shape == Z.shape)
    return A, cache


# ===================================================( Creating Minibatches )====================================================
def rand_mini_batches(X, Y, minibatch_size=64, seed=1):
    """Returns the minibatches of X and corresponding Y of the given size.
    
        Arguments:
            X (numpy.ndarray): Inputs Array.
            Y (numpy.ndarray): Output Labels.
            minibatch_size (int): Size of each minibatch.
            seed (int): Seed value for randomness.
        
        Returns:
            list: - **minibatches**: List of minibatches where each minibatch contains a minibatch of X and a minibatch of its corresponding Y.
            
        Examples:
            >>> X = np.random.randn(20,20)
            >>> Y = np.random.rand(1,20)
            >>> minibatches = rand_mini_batches(X,Y,minibatch_size = 4, seed = 2)
            >>> print(minibatches[0][0].shape)
            >>> print(minibatches[0][1].shape)
            
            Outputs:
                (20, 4)
                (1, 4)
    """
    classes = Y.shape[0]
    np.random.seed(seed)  # varying the seed value so that the minibatchs become random in each epoch
    m = X.shape[1]  # number of training examples
    minibatches = []

    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((classes, m))

    # Partition (shuffled_X, shuffled_Y) except for the last batch
    num_complete_minibatches = m // minibatch_size  # number of mini batches of size minibatch_size
    for k in range(0, num_complete_minibatches):
        minibatch_X = shuffled_X[:, k * minibatch_size: (k + 1) * minibatch_size]
        minibatch_Y = shuffled_Y[:, k * minibatch_size: (k + 1) * minibatch_size]
        minibatch = (minibatch_X, minibatch_Y)
        minibatches.append(minibatch)

    # Last batch (last minibatch <= minibatch_size)
    if m % minibatch_size != 0:
        minibatch_X = shuffled_X[:, num_complete_minibatches * minibatch_size: m]
        minibatch_Y = shuffled_Y[:, num_complete_minibatches * minibatch_size: m]
        minibatch = (minibatch_X, minibatch_Y)
        minibatches.append(minibatch)
        
    del shuffled_X, shuffled_Y #clearing unused memory

    return minibatches


# =====================================================( Time Conversion )======================================================
def convert_time(millisec):
    """Converts time in miliseconf to higher values.
    
        Arguments:
            milisec (int): Time in mili-seconds.
        
        Returns: 
            tuple: Following values
            - **hours** (int):  Time in hours.
            - **mins** (int): Time in minutes.
            - **secs** (int):  Time in seconds.
            - **milisec** (int): Time in mili-seconds.
            
        Example:
            >>> hr,mins,sec,milisec = convert_time(millisec = 12450)
            >>> print("%dhr %dmins %ds %dms"%(hr,mins,sec,milisec))
            
            Outputs:
                0hr 0mins 12s 450ms
    """
    # converting millisecons to hours, minutes, seconds and millisecond
    # the large numbers like 3.6e+6 comes from the relation between the time units
    hours = millisec // 3.6e+6
    mins = (millisec % 3.6e+6) // 60000
    secs = ((millisec % 3.6e+6) % 60000) // 1000
    millisec = ((millisec % 3.6e+6) % 60000) % 1000

    return (hours, mins, secs, millisec)


# =======================================( Computing and Visualizing Evaluation Matrices )=======================================
def confusion_matrix(y_orig, prediction):
    """Returns a confusion matrix for a given output labels and prediction.
    
        Arguments:
            y_orig (numpy.ndarray): Original Output Labels of shape(m,1); m = # of examples.
            prediction (numpy.ndarray): Predicted Labels of the dataset of shape (1,m).
        
        Returns:
            numpy.ndarray:- **cm**: 2D confusion matrix.
            
        Example:
            >>> y = np.array([[1,2,3,3,0,2,5,4,2,2]]).reshape(10,1)
            >>> pred = np.array([[2,2,1,3,0,2,5,4,2,2]]).reshape(1,10)
            >>> prediction = {"First Prediction":pred}
            >>> cm_train = confusion_matrix(y,prediction)
            >>> print(cm_train)
            
            Output:
                [[1 0 0 0 0 0]
                 [0 0 1 0 0 0]
                 [0 0 4 0 0 0]
                 [0 1 0 1 0 0]
                 [0 0 0 0 1 0]
                 [0 0 0 0 0 1]]
    """
    first_predict = prediction["First Prediction"]

    y_predicted = first_predict[0].T

    m = y_orig.shape[0]
    classes = len(np.unique(y_orig))  # or simply take classes = 10 for mnist or fashion-mnist

    cm = np.zeros((classes, classes))  # creating the matrix frame for the confusion matrix

    # generating the values in the confusion metrix
    for i in range(m):
        cm[y_orig[i], y_predicted[i]] += 1

    return cm.astype(int)


# ------------------------------------------------------------------------------------------------------------------------------
def plot_confusion_matrix(cm, dataset_type, dataset="mnist"):
    """Plots the Heatmap of the confusion matrix.
    
        Arguments:
            cm (numpy.ndarray): 2D confusion matrix.
            dataset_type (str): Type of dataset. May be training or dev or test.
            dataset (str): Dataset used to train the model. Default to 'mnist'
            
        Example:
            >>> plot_confusion_matrix(cm, dataset_type)
    """
    # plotting the metrix
    fig, ax = plt.subplots(figsize=(10, 10))
    im = plt.imshow(cm, cmap="GnBu")  # RdYlGn, PiYG, Accent,Blues,viridis, YlGnBu

    # plotting the color bar of the plot size
    fig.colorbar(im, ax=ax, fraction=0.045)

    if (len(dataset_type) != 0):
        visual_title = "Confusion Matrix for %s Set " % dataset_type.capitalize()
    else:
        raise ValueError("Dataset set must be training or dev or test set")

    # getting the label description
    label_desc = label_description(dataset)
    desc = [label_desc[i] for i in range(0, 10)]

    # annotating the plot
    ax.set_title(visual_title, fontsize=24, pad=20)
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xlabel("Predicted", fontsize=20)
    ax.set_ylabel("Expexted", fontsize=20)

    ax.set_xticklabels(desc)
    ax.set_yticklabels(desc)

    # setting horizontal axes labeling to top.
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # creating the threshold for color change in the visualization
    thres = cm.max() // 2

    for i in range(10):
        for j in range(10):
            # calculating the percentage of the total image denoted by the cell
            per = cm[i, j] / cm.sum() * 100
            # putting up the text inside the plot cells
            ax.text(j, i, "%d\n%.2f%%" % (cm[i, j], per),
                    ha="center", va="center", color="w" if cm[i, j] > thres else "black")

    fig.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------------------------------------------------------
def precision(label, cm):
    """Returns the precision for the prediction of an individual label.
    
        Arguments:
            label (int): unique labels (each class).
            cm (numpy.ndarray): 2D confusion matrix.
        
        Returns:
            numpy.float64: - **prec**: Precision for the given label.
            
        Example:
            >>> cm = [[1 0 0 0 0 0]
            ...       [0 0 1 0 0 0]
            ...       [0 0 4 0 0 0]
            ...       [0 1 0 1 0 0]
            ...       [0 0 0 0 1 0]
            ...       [0 0 0 0 0 1]]
            >>> label = 2
            >>> prec = precision(label, cm)
            >>> print(prec)
            
            Output:
                0.8
    """
    col = cm[:, label]  # selecting the True Positive and false positive values
    prec = cm[label, label] / col.sum()  # perc = TP / (TP + FP)

    return prec


# ------------------------------------------------------------------------------------------------------------------------------
def recall(label, cm):
    """Returns the recall for the prediction of an individual label.
    
        Arguments:
            label (int): unique labels (each class).
            cm (numpy.ndarray): 2D confusion matrix.
        
        Returns:
            numpy.float64: - **rec**: Recall for the given label.
            
        Example:
            >>> cm = [[1 0 0 0 0 0]
            ...       [0 0 1 0 0 0]
            ...       [0 0 4 0 0 0]
            ...       [0 1 0 1 0 0]
            ...       [0 0 0 0 1 0]
            ...       [0 0 0 0 0 1]]
            >>> label = 2
            >>> rec = recall(label, cm)
            >>> print(rec)
            
            Output:
                1.0
    """
    row = cm[label, :]  # selecting the True Positive and false Negative values
    rec = cm[label, label] / row.sum()  # rec = TP / (TP + FN)

    return rec


# ------------------------------------------------------------------------------------------------------------------------------
def f1_score(prec, rec):
    """Returns the f1-score for the prediction of an individual label.
    
        Arguments:
            prec (numpy.float64): precision for a label.
            rec (numpy.float64): recall for a label.
        
        Returns:
            numpy.float64: - **f1**: f1-score for the precision and recall.
            
        Example:
            >>> prec = 0.8
            >>> rec = 1.0
            >>> f1 = f1_score(prec,rec)
            >>> print(f1)
            
            Output:
                0.888888888888889
    """
    f1 = (2 * prec * rec) / (prec + rec)

    return f1


# ------------------------------------------------------------------------------------------------------------------------------
def macro_precision_average(precs):
    """Returns the macro average of the precisions for the prediction of all the label.
    
        Arguments:
            precs (list): lists of precisions of type numpy.float64.
        
        Returns:
            numpy.float64: - **prec_mac_avg**: macro average of the precisions.
            
        Example:
            >>> avg_precision = macro_precision_average(prec)
    """
    count = len(precs)
    prec_mac_avg = np.sum(precs) / count

    return prec_mac_avg


# ------------------------------------------------------------------------------------------------------------------------------
def macro_recall_average(recs):
    """Returns the macro average of the recall for the prediction of all the label.
    
        Arguments:
            recs (list): lists of recalls of type numpy.float64.
        
        Returns:
            numpy.float64: - **rec_mac_avg**: macro average of the recalls.
            
        Example:
            >>> avg_recall = macro_recall_average(rec)
    """
    count = len(recs)
    rec_mac_avg = np.sum(recs) / count

    return rec_mac_avg


# ------------------------------------------------------------------------------------------------------------------------------
def macro_f1_score(f1s):
    """Returns the macro average of the f1-score for the prediction of all the label.
    
        Arguments:
            f1s (list): lists of f1-scores of type numpy.float64.
        
        Returns:
            numpy.float64: - **f1_mac_avg**: macro average of the f1-scores.
            
        Example:
            >>> avg_f1 = macro_f1_score(f1)
    """
    count = len(f1s)
    f1_mac_avg = np.sum(f1s) / count

    return f1_mac_avg


# ------------------------------------------------------------------------------------------------------------------------------
def accuracy(cm):
    """Returns the accuracy of the prediction.
    
        Arguments:
            cm (numpy.ndarray): 2D confusion matrix.

        Returns:
            numpy.float64: - **acc**: Accuracy of the prediction.
            
        Example:
            >>> cm = [[1 0 0 0 0 0]
            ...       [0 0 1 0 0 0]
            ...       [0 0 4 0 0 0]
            ...       [0 1 0 1 0 0]
            ...       [0 0 0 0 1 0]
            ...       [0 0 0 0 0 1]]
            >>> acc = accuracy(cm)
            
            Outputs:
                0.8
    """
    diagonal_sum = cm.trace()  # getting the total truely classified images
    sum_of_all_elements = cm.sum()  # getting the total number of images
    acc = diagonal_sum / sum_of_all_elements

    return acc


# ------------------------------------------------------------------------------------------------------------------------------
def model_metrics(cm):
    """Returns the metrices and macro metrices for the evaluation of the model.
        Metrices includes: Precision, Recall and F1-score
        Macro Metrices includes: Macro Precision Average, Macro Recall Average and Macro F1-score Average
    
        Arguments:
            cm (numpy.ndarray): 2D confusion matrix.

        Returns: tuple: Following Values
            - **metrices** (dict): Model Metrices including list of Precision, Recall and F1-score.
            - **macro_metrices** (dict): Model Macro metrices including Macro Precision Average, Macro Recall Average and Macro F1-score Average.
            - **acc** (numpy.float64): Accuracy of the prediction.
            
        Example:
            >>> metrics, macro_metrics, acc = model_metrics(cm)
    """
    precs = []
    recs = []
    f1s = []
    metrics = {}
    macro_metrics = {}
    # calculating precision, recall and f1-score for all the classes or labels
    for label in range(10):
        precs.append(precision(label, cm))
        recs.append(recall(label, cm))
        f1s.append(f1_score(precs[label], recs[label]))

    # calculating the macro average metrices
    avg_precision = macro_precision_average(precs)  # calculating the macro metrices
    avg_recall = macro_recall_average(recs)
    avg_f1 = macro_f1_score(f1s)
    acc = accuracy(cm)

    metrics = {"Precision": precs,
               "Recall": recs,
               "F1-Score": f1s}
    macro_metrics = {"Precision": avg_precision,
                     "Recall": avg_recall,
                     "F1-Score": avg_f1}

    return metrics, macro_metrics, acc


# ------------------------------------------------------------------------------------------------------------------------------
def metric_summary(metrics, macro_metrics, accuracy):
    """Displays the metric summary after evaluation.
    
        Arguments:
            metrices (dict): Model Metrices including list of Precision, Recall and F1-score.
            macro_metrices (dict): Model Macro metrices including Macro Precision Average, Macro Recall Average and Macro F1-score Average.
            accuracy (numpy.float64): Accuracy of the prediction.
            
        Example:
            >>> metric_summary(metrics, macro_metrics, acc)
    """
    print("+===============+===============+===============+===============+")
    print("| Label \t| Precision \t| Recall \t| F1 Score \t|")
    print("+===============+===============+===============+===============+")
    prec = metrics["Precision"]
    rec = metrics["Recall"]
    f1 = metrics["F1-Score"]

    for label in range(len(prec)):
        print("| %d \t\t|  %.5f \t|  %.5f \t|  %.5f \t|" % (label, prec[label], rec[label], f1[label]))

    print("+===============+===============+===============+===============+")

    avg_precision = macro_metrics["Precision"]
    avg_recall = macro_metrics["Recall"]
    avg_f1 = macro_metrics["F1-Score"]
    acc = accuracy

    print("| Macro Avg \t|  %.5f \t|  %.5f \t|  %.5f \t|" % (avg_precision, avg_recall, avg_f1))
    print("+===============+===============+===============+===============+")

    print("\n Accuracy \t\t  %.5f" % (acc))


# =========================================( Visualizing Predictions and acc-loss plots )========================================
def visualize_training_results(train_accs, val_accs, train_loss, val_loss):
    """Visualize the traininig accuracy and loss, validation accuracy and loss over the training time.
    
        Arguments:
            train_accs (list): training accuracies obtained in all the minibatches in all epochs
            val_accs (list): validation accuracies obtained in all the minibatches in all epochs
            train_loss (list): training losses obtained in all the minibatches in all epochs
            val_loss (list): validation losses obtained in all the minibatches in all epochs
            
        Example:
            >>> visualize_training_results(train_accs, val_accs, train_losses, val_losses)    
    """
    # creating subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 15))
    fig.subplots_adjust(wspace=.2, hspace=.5)

    # plotting the loss
    axes[0].plot(np.squeeze(train_loss), label='Training Loss', color='blue')
    axes[0].plot(np.squeeze(val_loss), label='Validation Loss', color='red')
    axes[0].legend(loc='upper right')  # setting up legend location to upper right corner of the plot
    axes[0].set_title("Training and Validation Loss Graph", fontsize=16, pad=10)
    axes[0].set_xlabel("No. of Epochs", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_ylim(bottom=0)
    axes[0].grid(color='grey', alpha=0.5)

    # plotting the accuracy
    axes[1].plot(np.squeeze(train_accs), label='Training Accuracy', color='blue')
    axes[1].plot(np.squeeze(val_accs), label='Validation Accuracy', color='red')
    axes[1].legend(loc='lower right')  # setting up legend location to lower right corner of the plot
    axes[1].set_title("Training and Validation Accuracy Graph ", fontsize=16, pad=10)
    axes[1].set_xlabel("No. of Epochs", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_ylim(top=1)
    axes[1].grid(color='grey', alpha=0.5)

    plt.show()


# ------------------------------------------------------------------------------------------------------------------------------
def visualize_prediction(x_orig, y_orig, prediction, dataset_type, dataset="mnist"):
    """Displays 10 random images along with their true and predicted labels. 
        Both initial prediction and second guess are displayed.
    
        Arguments:
            x_orig (numpy.ndarray): original input data.
            y_orig (numpy.ndarray): original output labels.
            prediction (numpy.ndarray): predictions obtained after training the model.
            dataset_type (str): Type of dataset. May be training, dev  or test.
            dataset (str): Dataset used to train the model. Default to 'mnist'
            
        Example:
            >>> visualize_prediction(x_orig, y_orig.T, prediction, dataset_type = "training")
    """
    if (len(dataset_type) != 0):
        visual_title = "Sample %s Data Set " % dataset_type.capitalize()
    else:
        raise ValueError("Dataset set must be training or dev or test set")

    # getting the random index of 8 images to plot
    index = np.random.randint(0, 1000, 8)

    # getting the label description
    label_desc = label_description(dataset)

    # plotting the images along with the predictions
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(visual_title)

    first_lbl, first_prob = prediction["First Prediction"]
    sec_lbl, sec_prob = prediction["Second Prediction"]

    for ax, i in zip(axes.flatten(), index):
        ax.imshow(x_orig[i].squeeze(), interpolation='nearest', cmap='Greys')
        ax.set(title="True Label: %d | %s" % (y_orig[0, i], label_desc[y_orig[0, i]]))
        ax.set(xlabel="Prediction: %d | With Prob: %.4f \n2nd Guess: %d | With Prob: %.4f" % (
        first_lbl[0, i], first_prob[0, i], sec_lbl[0, i], sec_prob[0, i]))


# ------------------------------------------------------------------------------------------------------------------------------
def visualize_mislabelled_images(x_orig, y_orig, prediction, dataset_type, dataset="mnist"):
    """Displays 10 wrongly predicted images along with their true and predicted labels. 
        Both initial prediction and second guess are displayed.
    
        Arguments:
            x_orig (numpy.ndarray): original input data.
            y_orig (numpy.ndarray): original output labels.
            prediction (numpy.ndarray): predictions obtained after training the model.
            dataset_type (str): Type of dataset. May be training, dev  or test.
            dataset (str): Dataset used to train the model. Default to 'mnist'
            
        Example:
            >>> visualize_mislabelled_images(x_orig, y_orig.T, prediction, dataset_type = "training")
    """
    first_lbl, first_prob = prediction["First Prediction"]
    sec_lbl, sec_prob = prediction["Second Prediction"]

    true_prediction = np.equal(first_lbl, y_orig)
    mislabelled_indices = np.asarray(np.where(true_prediction == False))
    print("Total Mislabelled Images: " + str(len(mislabelled_indices[0])))

    if (len(dataset_type) != 0):
        visual_title = "Sample Mislabelled %s Set Images " % dataset_type.capitalize()
    else:
        raise ValueError("Dataset set must be training or dev or test set")

    # getting the label description
    label_desc = label_description(dataset)

    # plotting the mislabelled images along with the predictions
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 8))
    fig.subplots_adjust(hspace=1)
    fig.suptitle(visual_title)

    for ax, i in zip(axes.flatten(), mislabelled_indices[1]):
        ax.imshow(x_orig[i].squeeze(), interpolation='nearest', cmap="Greys")
        ax.set(title="True Label: %d | %s" % (y_orig[0, i], label_desc[y_orig[0, i]]))
        ax.set(xlabel="Prediction: %d | With Prob: %.4f \n2nd Guess: %d | With Prob: %.4f" % (
        first_lbl[0, i], first_prob[0, i], sec_lbl[0, i], sec_prob[0, i]))


# =================================================( Saving and Loading Model )==================================================
def save_model(file_name, model):
    """Saves the parameters of the trained model into a pickle file.
    
        Arguments:
            file_name (str): name of the file to be saved.
            model (dict): trained model to be saved in the file. Consists of Parameters and activations 
            
        Example:
            >>> save_model(file_name = path + fname, model)
    """
    with open(file_name, 'wb') as output_file:
        pickle.dump(model, output_file)


# ------------------------------------------------------------------------------------------------------------------------------
def load_model(file_name):
    """Load the saved model from a pickle file
    
        Arguments:
            file_name (str): name of the file to be loaded from
        
        Returns:
            model (dict): trained model to be saved in the file. Consists of Parameters and activations 
            
        Example:
            >>> model = load_model(file_name = path + fname)
    """
    try:
        with open(file_name, 'rb') as input_file:
            model = pickle.load(input_file)

        return model

    except(OSError, IOError) as e:
        print(e)

# ==============================================================================================================================
