import numpy as np
import tensorflow as tf
from metrics import dice_coef, sensitivity, specificity, precision

#Layers based on VGG structure
#Filters/biases in VGG come from Numpy file
#Pseudo/skeleton code

class PseudoVGG(tf.keras.Model):
    def __init__(self):
        super(PseudoVGG, self).__init__()
        #Hyperparams
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.batch_size = 32
        self.epochs = 15
        self.color = 'RGB' #should be 'RGB' or 'L'
        kernel_size_1 = 3
        kernel_size_2 = 2
        

        #Model Architecture
        self.conv1_1 = tf.keras.layers.Conv2D(64,kernel_size_1,activation='relu', padding='SAME',use_bias=True,bias_initializer='random_normal')
        self.conv1_2 = tf.keras.layers.Conv2D(64,kernel_size_1,activation='relu', padding='SAME',use_bias=True,bias_initializer='random_normal')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2), padding='SAME')
        
        self.conv2_1 = tf.keras.layers.Conv2D(128,kernel_size_1,activation='relu', padding='SAME',use_bias=True,bias_initializer='random_normal')
        self.conv2_2 = tf.keras.layers.Conv2D(128,kernel_size_1,activation='relu', padding='SAME',use_bias=True,bias_initializer='random_normal')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2), padding='SAME')
        
        self.conv3_1 = tf.keras.layers.Conv2D(256,kernel_size_1,activation='relu', padding='SAME',use_bias=True,bias_initializer='random_normal')
        self.conv3_2 = tf.keras.layers.Conv2D(256,kernel_size_1,activation='relu', padding='SAME',use_bias=True,bias_initializer='random_normal')
        self.conv3_3 = tf.keras.layers.Conv2D(256,kernel_size_1,activation='relu', padding='SAME',use_bias=True,bias_initializer='random_normal')
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2), padding='SAME')
            
        self.conv4_1 = tf.keras.layers.Conv2D(512,kernel_size_2,activation='relu', padding='SAME',use_bias=True,bias_initializer='random_normal')
        self.conv4_2 = tf.keras.layers.Conv2D(512,kernel_size_2,activation='relu', padding='SAME',use_bias=True,bias_initializer='random_normal')
        self.conv4_3 = tf.keras.layers.Conv2D(512,kernel_size_2,activation='relu', padding='SAME',use_bias=True,bias_initializer='random_normal')
        self.pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2), padding='SAME')
            
        self.conv5_1 = tf.keras.layers.Conv2D(512,kernel_size_2, activation='relu', padding='SAME',use_bias=True,bias_initializer='random_normal')
        self.conv5_2 = tf.keras.layers.Conv2D(512,kernel_size_2, activation='relu', padding='SAME',use_bias=True,bias_initializer='random_normal')
        self.conv5_3 = tf.keras.layers.Conv2D(512,kernel_size_2, activation='relu', padding='SAME',use_bias=True,bias_initializer='random_normal')
        self.pool5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2), padding='SAME')
        self.flatten = tf.keras.layers.Flatten()
            
        self.dense6 = tf.keras.layers.Dense(4096, use_bias=True, activation='relu', bias_initializer='random_normal')
        self.dense7 = tf.keras.layers.Dense(1000, use_bias=True, activation='relu', bias_initializer='random_normal')
        self.dense8 = tf.keras.layers.Dense(2, use_bias=True, activation=None, bias_initializer='random_normal')
    
    def call(self, covid_input):
        """
        :param covid_inputs: Tensor or Numpy Array
            input images all of shape (batch_size, imsize, imsize, channels)
        :return: Tensor
            prediction values of chape (batch_size, 2)
        """
        conv1_1 = self.conv1_1(covid_input)
        conv1_2 = self.conv1_2(conv1_1)
        pool1 = self.pool1(conv1_2)
        
        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        pool2 = self.pool2(conv2_2)
    
        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_2 = self.conv3_2(conv3_2)
        pool3 = self.pool3(conv3_2)   

        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        conv4_2 = self.conv4_2(conv4_2)
        pool4 = self.pool4(conv4_2)  

        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)
        conv5_3 = self.conv5_3(conv5_2)
        pool5 = self.pool5(conv5_3)
        flat = self.flatten(pool5)

        dense6 = self.dense6(flat)
        dense7 = self.dense7(dense6)
        dense8 = self.dense8(dense7)
        
        probs = tf.nn.softmax(dense8)        
        return probs

    def loss_function(self, y_true, y_pred):
        """
        :param y_true:Tensor - shape (batch_size, 2)
            Truth labels one hot encoded
        :param y_pred:Tensor - shape (batch_size, 2)
            prediction value probabilities 
        :return: Tensor - single float value 
            binary crossentropy 
        """
        crossentropy = tf.math.reduce_sum(tf.keras.losses.binary_crossentropy(y_true, y_pred))
        #tf.keras.losses.binary_crossentropy(y_true, y_pred)
        #tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        return crossentropy 
        #tried all the below with reduce mean instead of reduce sum
        #but need to take argmax first for all but dice 
        #(as they dont take in one-hot encoded inputs)
        #- dice_coef(y_true, y_pred)
        #- sensitivity(y_true, y_pred)
        #- specificity(y_true, y_pred)
        #- precision(y_true, y_pred)



