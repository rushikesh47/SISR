
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 02 10:18:31 2018

@author: Rushikesh
"""

from __future__ import print_function
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import h5py
import math
import cv2

def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    
    #return -10. * (np.log10(K.mean(K.square(y_pred - y_true))))
    return 10.0 * K.log(1.0 / (K.mean(K.square(y_pred - y_true)))) / K.log(10.0)


"""
Step Decay is the function used to modify the learning rate of the optimization algorithm.
Initial learning rate is kept at 0.001 and after every 20 iterations, the learning 
rate is reduced by a factor of 0.1 as defined below
"""
   


def step_decay(epoch):
   	initial_lrate = 0.001
   	drop = 0.1
   	epochs_drop = 20
   	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
   	return lrate
   

#paramaters
    
#Number of images fed to cnn at a time for training.
#Generally kept a power of 2, to fully utilize a processor's core processing power
batch_size = 64

#Number of backpropagatio iterations to optimize the weights
nb_epoch = 10000


#image dimensions

#Input image dimensions
img_rows, img_cols = 33, 33

#Output image dimensions
out_rows, out_cols = 33, 33


#Number of filters in respective convolutional layers
n1 = 64
n2 = 32
n3 = 33


#Respective convolutional layer filter sizes [f * f]
f1 = 9
f2 = 1
f3 = 5


#loading training data from h5 file.
#This h5 file is the output of preprocessing code.
#in_train is input data to cnn
#out_train is output image labels.
file = h5py.File('train/Train/train91_mscale.h5', 'r')
in_train = file['data'][:]
out_train = file['label'][:]


#File close
file.close()


#load validation \ testing data
file = h5py.File('train/Test/test_mscale.h5', 'r')
in_test = file['data'][:]
out_test = file['label'][:]


#File close
file.close()


#convert data into 'float32' data format
in_train = in_train.astype('float32')
out_train = out_train.astype('float32')
in_test = in_test.astype('float32')
out_test = out_test.astype('float32')

#Converting the data into 'Channel first' format if it is in 'Channel last' format
#here data is in 4D array with the dimension [n, ch, r, c]
#n = number of patches
#ch = number of channels = 1 in our case
#r,c= number of rows and columns respectively
if K.image_dim_ordering() == 'th':
    in_train = in_train.reshape(in_train.shape[0], 1, img_rows, img_cols)
    in_test  = in_test.reshape(in_test.shape[0], 1, img_rows, img_cols)
    out_train = out_train.reshape(out_train.shape[0], 1, out_rows, out_cols)
    out_test = out_test.reshape(out_test.shape[0], 1, out_rows, out_cols)
    input_shape = (1, img_rows, img_cols)


#printing number of training / validation patches
print('in_train shape:', in_train.shape)
print(in_train.shape[0], 'train samples')
print(in_test.shape[0], 'test samples')



#Defining SR Model Architecture
def model(input_shape):
    x = Input(input_shape)
    c1 = Conv2D(n1, (f1,f1), activation = 'relu', init = 'he_normal', border_mode='same', name = 'layer1')(x)
    c2 = Conv2D(n2, (f2, f2), activation = 'relu', init = 'he_normal', border_mode='same', name = 'layer2')(c1)
    c3 = Conv2D(n3, (f3, f3), init = 'he_normal', border_mode='same', name = 'layer3')(c2)
    model = Model(inputs = x, outputs = c3)
    return model


#compile
model = model(in_train.shape[1:])
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8) 
model.compile(loss='mse', metrics=[PSNRLoss], optimizer=adam)

   
# learning schedule callback
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]
history = model.fit(in_train, out_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks = [lrate],
          verbose=1, validation_data=(in_test, out_test))            
print(history.history.keys())


#save model and weights
json_string = model.to_json()  
open('srcnn_model.json','w').write(json_string)  
model.save_weights('srcnn_model_weights1.h5') 


# summarize history for loss
plt.plot(history.history['PSNRLoss'])
plt.plot(history.history['val_PSNRLoss'])
plt.title('model loss')
plt.ylabel('PSNR/dB')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()


#bicubic_img = cv2.resize(cv2.resize(img,None, fx = 1/3, fy = 1/3, interpolation = cv2.INTER_CUBIC),None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
#10 * (np.log10(1/(np.mean(np.square(cv2.resize(cv2.resize(in_train[90,0,:,:],None, fx = 1/11, fy = 1/11, interpolation = cv2.INTER_CUBIC),None, fx = 11, fy = 11, interpolation = cv2.INTER_CUBIC) - in_train[90,0,:,:])))))
"""
x = history.history['PSNRLoss']
for i in range(100):
    x[i] = 10 * (np.log10(1/(np.mean(np.square(cv2.resize(cv2.resize(in_train[i,0,:,:],None, fx = 1/11, fy = 1/11, interpolation = cv2.INTER_CUBIC),None, fx = 11, fy = 11, interpolation = cv2.INTER_CUBIC) - in_train[i,0,:,:])))))
plt.plot(x)
plt.plot(history.history['val_PSNRLoss'])
plt.title('method comparison')
plt.ylabel('PSNR/dB')
plt.xlabel('epoch')
plt.legend(['Interpolation', 'CNN'], loc='lower right')
plt.show()
"""