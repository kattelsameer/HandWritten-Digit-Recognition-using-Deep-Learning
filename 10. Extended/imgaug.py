#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 22:13:40 2020

@author: befrenz
"""
# ===================================================( Loading Dependencies )===================================================
# standard python modules
import os # for working with directories
import os.path # for working with paths

#core python modules
from PIL import Image # for loading and saving images
from scipy import ndimage # for core image processing namely, rotate, blur and shift

import numpy as np # for all other maths operations 

#custom project modules
from dataset import get_files 
from dataset import sample_dataset


# =====================================================( Rotate Images )=====================================================
#rotating and rotating+zooming

def rotate_images(images,labels, save_image = False):
    m = labels.shape[0]
    
    if save_image:
        path = "dataset/mnist_augmented/rotated/"
        if not os.path.exists(path):
            os.makedirs(path)  # creating required directories recursively

    rotated_images = []
    
    for i in range(m):    
        # randomly selecting angle between the range -45 to -15 and 15 to 45
        pos_angle = np.random.randint(low = 10, high = 45)
        neg_angle = np.random.randint(low = -45, high = -10)
        angle = np.random.choice([pos_angle,neg_angle])

        rotated_img = ndimage.rotate(images[i], angle, reshape=False, mode = "nearest")
        
        if(save_image):
            img = Image.fromarray((rotated_img * 255).astype(np.uint8))
            img.save(path + str(np.squeeze(labels[i]))+"_rotated_"+str(i+1)+".jpg")
            
        rotated_images.append(rotated_img)
    
    return np.asarray(rotated_images), labels

# =====================================================( Blur Images )=====================================================
#blurring using Gaussian Filter
def blur_images(images, labels, filter_mode = "random", random_filter = False, save_image = False):
    m = labels.shape[0]
    
    if save_image:
        path = "dataset/mnist_augmented/blurred/"
        if not os.path.exists(path):
            os.makedirs(path)  # creating required directories recursively

    blurred_images = []

    for i in range(m):    
        
        if random_filter:
            filters = ['gaussian', 'maximum', 'minimum', 'median', 'uniform']
            filter_mode = np.random.choice(filters)
            
        if filter_mode == "gaussian":
            sig = np.random.uniform(low = 0, high = 2)
            blurred_img = ndimage.gaussian_filter(images[i], sigma=sig,  mode = "nearest")

        elif filter_mode =="maximum":
            s = np.random.uniform(low = 2, high = 4)
            blurred_img = ndimage.maximum_filter(images[i], size= s,  mode = "nearest")

        elif filter_mode == "minimum":
            s = np.random.uniform(low = 0, high = 4)
            blurred_img = ndimage.minimum_filter(images[i], size= s,  mode = "nearest")

        elif filter_mode == "median":
            s = np.random.randint(low = 2, high = 6)
            blurred_img = ndimage.median_filter(images[i], size= s,  mode = "nearest")

        elif filter_mode == "uniform":
            s = np.random.uniform(low = 2, high = 6)
            blurred_img = ndimage.uniform_filter(images[i], size= s, mode = "nearest")

        else:
            raise ValueError("filter mode should only be 'gaussian', 'maximum', 'minimum', 'median', or 'uniform'")
        
        if(save_image):
                img = Image.fromarray((blurred_img * 255).astype(np.uint8))
                img.save(path + str(np.squeeze(labels[i]))+"_blurred_"+filter_mode+"_"+str(i+1)+".jpg")

        blurred_images.append(blurred_img)
    
    return np.asarray(blurred_images), labels

# =====================================================( Shift Images )=====================================================
#Shifting image
def shift_images(images,labels, shifting ="both", save_image = False):
    m = labels.shape[0]
    
    if save_image:
        path = "dataset/mnist_augmented/shifted/"
        if not os.path.exists(path):
            os.makedirs(path)  # creating required directories recursively

    shifted_images = []
    
    for i in range(m):    
         # randomly selecting angle between the range -45 to -15 and 15 to 45
        
        if shifting == "horizontal":
            xs = 0
            ys = np.random.uniform(low = -6, high = 6)
            
            
        elif shifting == "vertical":
            ys = 0
            xs = np.random.uniform(low = -6, high = 6)
            
        elif shifting == "both":
            xs = np.random.uniform(low = -6, high = 6)
            ys = np.random.uniform(low = -6, high = 6)
        else:
            raise ValueError("Shifting should only be 'horizontal', 'vertical', or 'both'")
            

        shifted_img = ndimage.shift(images[i], shift= (xs,ys), mode = "nearest")
        
        if(save_image):
            img = Image.fromarray((shifted_img * 255).astype(np.uint8))
            img.save(path + str(np.squeeze(labels[i]))+"_shifted_"+shifting+"_"+str(i+1)+".jpg")
            
        shifted_images.append(shifted_img)
    
    return np.asarray(shifted_images), labels

# =====================================================( Crop and Pad Images )=====================================================
#cropping image
def crop_and_pad_images(images, labels, crop_center = False, save_image = False):
   
    m = labels.shape[0]
    
    if save_image:
        path = "dataset/mnist_augmented/cropped_and_padded/"
        if not os.path.exists(path):
            os.makedirs(path)  # creating required directories recursively

    cropped_images = []
    
    for i in range(m):    
        lx, ly = images[i].shape
        cropped_image = np.zeros((lx,ly))
        cropped_image.fill(0) #padding with pixelvalue; 255 for white and 0 for black padding
        if crop_center: # this will work as zoom and crop
            # randomly selecting  the range for central cropping
            c = np.random.randint(low = 4, high = 10)
            cropped_image[lx // c: - lx // c, ly // c: - ly // c] = np.copy(images[i][lx // c: - lx // c, ly // c: - ly // c])
        else: 
            # randomly selecting  the range for cropping across different axis
            clx = np.random.randint(low = 4, high = 10)
            crx= np.random.randint(low = 4, high = 10)
            cly = np.random.randint(low = 4, high = 10)
            cry = np.random.randint(low = 4, high = 10)
            
            cropped_image[lx // clx: - lx // crx, ly // cly: - ly // cry] = np.copy(images[i][lx // clx: - lx // crx, ly // cly: - ly // cry])

        if(save_image):
            img = Image.fromarray((cropped_image * 255).astype(np.uint8))
            img.save(path + str(np.squeeze(labels[i]))+"_cropped_"+str(i+1)+".jpg")
            
        cropped_images.append(cropped_image)
    
    return np.asarray(cropped_images), labels

# =====================================================( Flip Images Horizontally )=====================================================
#Flipping image
def horizontal_flip_images(images,labels, save_image = False):
    m = labels.shape[0]
    
    if save_image:
        path = "dataset/mnist_augmented/h_flipped/"
        if not os.path.exists(path):
            os.makedirs(path)  # creating required directories recursively

    flipped_images = []
    
    for i in range(m):    
        flip_h = np.fliplr(images[i])

        if(save_image):
#             https://stackoverflow.com/questions/55319949/pil-typeerror-cannot-handle-this-data-type
            img = Image.fromarray((flip_h * 255).astype(np.uint8))
            img.save(path + str(np.squeeze(labels[i]))+"_hFlipped_"+str(i+1)+".jpg")
            
        flipped_images.append(flip_h)
    
    return np.asarray(flipped_images), labels

# =====================================================( Augment Images )=====================================================
def augment_img(images_orig, labels, horizontal_flip = False, crop_and_pad = False, rotate = False, shift = False, blur = False, save_images = False, include_original = False):
    
    m = labels.shape[0]
    augmented_images = np.copy(images_orig)
    augmented_labels = np.copy(labels)

    #horizontal flipping
    if horizontal_flip:
        flipped_images, flipped_labels = horizontal_flip_images(images_orig, labels, save_image = save_images)
        augmented_images = np.concatenate((augmented_images, flipped_images), axis = 0)
        augmented_labels = np.concatenate((augmented_labels, flipped_labels), axis = 0)
     
    #random cropping and padding
    if crop_and_pad:
        cropped_images, cropped_labels = crop_and_pad_images(images_orig, labels, save_image = save_images)
        augmented_images = np.concatenate((augmented_images, cropped_images), axis = 0)
        augmented_labels = np.concatenate((augmented_labels, cropped_labels), axis = 0)
     
    #random rotating
    if rotate:
        rotated_images, rotated_labels = rotate_images(images_orig, labels, save_image = save_images)
        augmented_images = np.concatenate((augmented_images, rotated_images), axis = 0)
        augmented_labels = np.concatenate((augmented_labels, rotated_labels), axis = 0)
     
    #random shifting
    if shift:
        shifted_images, shifted_labels = shift_images(images_orig,labels, shifting ="both", save_image = save_images)
        augmented_images = np.concatenate((augmented_images, shifted_images), axis = 0)
        augmented_labels = np.concatenate((augmented_labels, shifted_labels), axis = 0)
    
    #random blurring
    if blur:
        blurred_images, blurred_labels = blur_images(images_orig, labels, random_filter = True, save_image =save_images)
        augmented_images = np.concatenate((augmented_images, blurred_images), axis = 0)
        augmented_labels = np.concatenate((augmented_labels, blurred_labels), axis = 0)
    
    
    #suffeling all the images
    augmented_images, augmented_labels = sample_dataset(augmented_images, augmented_labels, size_in_per = 100)
    
    if include_original:
        return augmented_images, augmented_labels
    else:
        return augmented_images[m::], augmented_labels[m:]
    
    
# =====================================================( Load Files from Directory )=====================================================


def load_images_from_file(path):
    #checking for the validity of the path
    if not os.path.exists(path):
            raise ValueError("Given folder doesnot exist")

    image_names = sorted(get_files(path))
    images = []
    lbls = []
    for image_name in image_names:
        fname = path + image_name
        image_data = np.asarray(Image.open(fname).resize((28,28)).convert('L')).reshape(28,28)
        if image_data[1,1] > 250: #if background is white, reversing the fore and background color
            image_data = 255 - image_data
        images.append(image_data.tolist())
        lbls.append(image_name[0])

    real_images = np.asarray(images)
    labels = np.asarray(lbls)
    
    return real_images, labels



# =====================================================( Download Dataset )=====================================================

