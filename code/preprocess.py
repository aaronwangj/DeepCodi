import pickle
import numpy as np
import tensorflow as tf
import os
import glob

from PIL import Image
from balanced_gen import BalancedDataGenerator

def normalize_image(image):
    image = image/np.max(image)
    """dividing by 255 (max value possible) is technically more correct 
    for image normalizations but runs into problems if you accidentally 
    double normalize so we kept it the other way"""
    #image = image/255.0
    return image


def get_balanced_data(path, imsize=224, batch_size=32, color='L'):
    """
    :param path: String
        relative path to the data folder something 
        like '../data/main_dataset/train/'
    :param imsize: Integer
        size of input image - image comes out as (imsize x imsize x channels)
    :param batch_size: Integer
        number of images in each batch
    :param color: Stinge of values 'L' or 'RGB'
        color type of input image. L is black and white with 1 channel 
        and 'RGB' is color with 3 channels
    :return: BalancedDataGenerator
        a datagenerator which runs preprocessing and returns batches accessed
        by integers indexing (i.e. generator[0] returns the first batch of inputs 
        and labels)
    """
    inputs, labels =  get_data_main(path, imsize=imsize, color=color, normalize=False)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        brightness_range=[0.5, 1.25],
        preprocessing_function=normalize_image if color == 'L' else tf.keras.applications.vgg16.preprocess_input
    )
        
    seed = 1

    #Feed Training data and training data generator into Balanced Data Generator: augments data such that it is not heavily imbalanced
    balanced_gen = BalancedDataGenerator(inputs, labels, datagen, batch_size=batch_size)

    return balanced_gen


def get_data_main(path, imsize=224, oversample=1, color='RGB', normalize=True):
    """
    Given a file path, returns an array of normalized inputs (images) and an array of 
    one_hot encoded binary labels. 

    :param path: String
        relative path to the data folder something 
        like '../data/main_dataset/train/'
    :param imsize: Integer
        size of input image - image comes out as (imsize x imsize x channels)
    :param batch_size: Integer
        number of images in each batch
    :param color: Stinge of values 'L' or 'RGB'
        color type of input image. L is black and white with 1 channel 
        and 'RGB' is color with 3 channels
    :param normalize: Boolean
        whether or not to run normalization. If running into BalancedDataGen
        it takes care of that for you, so you don't to double normalize
    :return: Numpy Array, Numpy Array
        a array containing the input images of desired size and type and 
        an array containing the labels one-hot encoded
     """
    covid_pics = glob.glob(path+"1_covid/*")
    if 'test' in path:
        non_covid_pics = glob.glob(path+"0_non/**/*")
    else:
        non_covid_pics = glob.glob(path+"0_non/*")
    num_pics = len(covid_pics)*oversample+len(non_covid_pics)
    if color == 'L':
        data = np.empty((num_pics, imsize, imsize, 1))
    else:
        data = np.empty((num_pics, imsize, imsize, 3))
    labels = np.zeros((num_pics, 2))
    index = 0
    for i in range(oversample):
        for pic in covid_pics:
            image = Image.open(pic).resize((imsize,imsize)).convert(color)
            im_data = np.asarray(image)
            if color == 'L':
                if normalize:
                    data[index] = np.expand_dims(normalize_image(im_data), -1)
                else:
                    data[index] = np.expand_dims(im_data, -1)
            else:
                data[index] = normalize_image(im_data) if normalize else im_data
            labels[index,1] = 1
            index += 1
    for pic in non_covid_pics:
        image = Image.open(pic).resize((imsize,imsize)).convert(color)
        im_data = np.asarray(image)
        if color == 'L':
            if normalize:
                data[index] = np.expand_dims(normalize_image(im_data), -1)
            else:
                data[index] = np.expand_dims(im_data, -1)
        else:
            data[index] = normalize_image(im_data) if normalize else im_data
        labels[index,0] = 1
        index += 1

    return data, labels


if __name__ == '__main__':
    path1 = '../data/kaggle_dataset/train/*'
    path2 = '../data/main_dataset/train/'
    data, labels = get_data_main(path2)
