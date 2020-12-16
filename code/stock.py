import numpy as np
import tensorflow as tf
import os

from tensorflow.keras.applications.vgg16 import preprocess_input,decode_predictions
from tensorflow.keras.callbacks import Callback
from balanced_gen import BalancedDataGenerator
from metrics import *
import metrics

from preprocess import get_data_main
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices = ['train','test'],default = 'train')
parser.add_argument('--weights',type=str,default = None)
args = parser.parse_args()


def train(model,train_data,train_labels):
    """
    Trains the model
    :param train_data: The training data returned by preprocessing.get_data()
    :param train_labels: The training labels returned by preprocessing.get_data()
    
    Augmentation is performed due to an imbalanced dataset. The data is oversampled and augmented utilizing the datagen params and balanced_gen.py
    """
    #Create Training Data Generator for augmentation
    CSVLogger = tf.keras.callbacks.CSVLogger('../results/stock/train_logs.csv',separator=",")
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        brightness_range=[0.5, 1.25],
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input
        )
        
    seed = 1

    #Feed Training data and training data generator into Balanced Data Generator: augments data such that it is not heavily imbalanced
    balanced_gen = BalancedDataGenerator(train_data, train_labels, train_datagen,batch_size = 32)
 
    train_steps = balanced_gen.steps_per_epoch
    
    #Stop Early if val_accuracy no longer improving
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        mode='min',
        patience=5
        )
        
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='../weights/stock/weights.{epoch:02d}-{loss:.2f}.hdf5',
        save_weights_only=True,
        save_freq = 'epoch',
        monitor='loss',
        mode='min',
        save_best_only=True)
        
    #Fit Model
    model.fit_generator(
        balanced_gen,
        steps_per_epoch=train_steps,
        epochs=10,
        callbacks=[early_stopping,checkpoint_callback,CSVLogger]
        )   
     
    
def test(model,test_path):
    """
    Tests the model
    :param test_path: The path to the test data
    """
    seed = 1

    #Create Test Generator
    testing_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input
        )
    testing_generator =testing_datagen.flow_from_directory(
        test_path,
        color_mode='rgb',
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical',
        seed=seed
        ) 
    test_steps = testing_generator.n//testing_generator.batch_size
    
    model.evaluate_generator(testing_generator,
    steps=test_steps,
    verbose=1)        

    
    
  
    
def main():
    """
    To run in testing mode, include following args:
    --mode test
    --weights path_to_weights + filename
    
    ex: python main.py --mode test --weights weights.10-0.09.hdf5
    """
    train_path = '../data/main_dataset/train/'
    test_path ='../data/main_dataset/test/'
    

    print("Loading the data...")
    
    #Load Training Data
    train_data, train_labels = get_data_main(train_path)
    print(len(train_data),len(train_labels))
    sensitivity_15 = tf.keras.metrics.Recall(.15,class_id = 1,name = "sensitivity_15")
    sensitivity_3 = tf.keras.metrics.Recall(.3,class_id = 1,name = "sensitivity_3")
    sensitivity_5 = tf.keras.metrics.Recall(.5,class_id = 1,name = "sensitivity_5")
    
    specificity_15 = tf.keras.metrics.Recall(.15,class_id = 0,name = "specificity_15")
    specificity_3 = tf.keras.metrics.Recall(.3,class_id = 0,name = "specificity_3")
    specificity_5 = tf.keras.metrics.Recall(.5,class_id = 0,name = "specificity_5")   
    
    print("Generating the model...")
    shape = (224, 224, 3)
    model = tf.keras.applications.VGG16(input_shape=shape, include_top=True,weights = None,classes = 2)
    model.compile(optimizer=tf.optimizers.Adam(.0001), loss='binary_crossentropy',run_eagerly=True, metrics=[dice_coef,sensitivity_15,sensitivity_3,sensitivity_5,specificity_15,specificity_3,specificity_5,tf.keras.metrics.Precision()])
    model.summary()
     
     
     
     
    # #What does VGG16 think these are with its own classifiers 
    # test_data, _ = get_data_main(train_path)
    # test_data = np.expand_dims(test_data[0], axis=0)
    # test_data =  preprocess_input(test_data)
    # preds = model.predict(test_data)
    # decoded = decode_predictions(preds)
    # print(decoded)
    if args.mode == 'train':
        print("Training...")
        train(model,train_data,train_labels)    
    else:
        print("Loading Weights...")
        model.load_weights(args.weights)
        print("Testing...")
        test(model,test_path)




if __name__ == '__main__':
    main()

