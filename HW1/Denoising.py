# -*- coding: utf-8 -*-
"""
The file contains what I have implemented for the bonus part about denoising 
of the first tp of the deep learning class.
@author: Pirashanth
"""
from mp1 import *
import matplotlib.pyplot as plt
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Convolution2D,MaxPooling2D,Flatten,BatchNormalization,Dropout

from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import UpSampling2D

############6 Bonus Question#############



#This first part contains the denoising tool dealed as a regression problem
#MSE loss
mcp = ModelCheckpoint('weights.best_hourglass.hdf5', monitor="val_loss",
                      save_best_only=True, save_weights_only=False)


X_train_noise,Y_train_noise = generate_dataset_noise(5000) 
X_test_noise,Y_test_noise = generate_test_set_noise() 


X_train_noise = X_train_noise.reshape((-1,72,72,1))
Y_train_noise = Y_train_noise.reshape((-1,72,72,1))

X_test_noise = X_test_noise.reshape((-1,72,72,1))
Y_test_noise = Y_test_noise.reshape((-1,72,72,1))

model =Sequential()

model.add(Convolution2D(60,kernel_size=(3,3),input_shape=(72,72,1,),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(40,kernel_size=(2,2),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(60,kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(40,kernel_size=(2,2),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(60,kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(20,kernel_size=(2,2),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(UpSampling2D(size=(2,2)))
model.add(Convolution2D(40,kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(20,kernel_size=(2,2),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(UpSampling2D(size=(2,2)))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(1,kernel_size=(3,3)))

model.compile(loss='mean_squared_error',
              optimizer='adam')

history= model.fit(X_train_noise,Y_train_noise,validation_split=0.1,epochs=50,batch_size=100,callbacks=[mcp])

model = keras.models.load_model('weights.best_hourglass.hdf5')

print('Model outcome regression:',model.evaluate(X_test_noise,Y_test_noise))

Y_pred_without_noise = model.predict(X_test_noise)

#print some results
plt.imshow(Y_train_noise[0].reshape(72,72), cmap='gray')
plt.title('Output train denoising')
plt.savefig('output_train.png', dpi=500)

plt.imshow(X_train_noise[0].reshape(72,72), cmap='gray')
plt.title('Input train denoising')
plt.savefig('input_train.png', dpi=500)

plt.imshow(Y_pred_without_noise[0].reshape(72,72), cmap='gray')
plt.title('Predicted Output test denoising')
plt.savefig('predict_output_test.png', dpi=500)

plt.imshow(X_test_noise[0].reshape(72,72), cmap='gray')
plt.title('Input test denoising')
plt.savefig('input_test.png', dpi=500)

#This second part contains the network that deals with our denoising
#task as a segmentation problem

mcp = ModelCheckpoint('weights.best_hourglass.hdf5', monitor="val_loss",
                      save_best_only=True, save_weights_only=False)


X_train_noise,Y_train_noise = generate_dataset_noise_segmentation(5000) 
X_test_noise,Y_test_noise = generate_test_set_noise_segmentation() 


X_train_noise = X_train_noise.reshape((-1,72,72,1))
Y_train_noise = Y_train_noise.reshape((-1,72,72,1))

X_test_noise = X_test_noise.reshape((-1,72,72,1))
Y_test_noise = Y_test_noise.reshape((-1,72,72,1))

mcp = ModelCheckpoint('weights.best_hourglass_segment.hdf5', monitor="val_loss",
                      save_best_only=True, save_weights_only=False)

model =Sequential()

model.add(Convolution2D(60,kernel_size=(3,3),input_shape=(72,72,1,),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(40,kernel_size=(2,2),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(60,kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(40,kernel_size=(2,2),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(60,kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(20,kernel_size=(2,2),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(UpSampling2D(size=(2,2)))
model.add(Convolution2D(40,kernel_size=(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(20,kernel_size=(2,2),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(UpSampling2D(size=(2,2)))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(1,kernel_size=(3,3)))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer='adam')

history= model.fit(X_train_noise,Y_train_noise,validation_split=0.1,epochs=50,batch_size=100,callbacks=[mcp])

model = keras.models.load_model('weights.best_hourglass_segment.hdf5')

print('Model outcome',model.evaluate(X_test_noise,Y_test_noise))

Y_pred_without_noise = model.predict(X_test_noise)

#print outcomes
plt.imshow(Y_train_noise[0].reshape(72,72), cmap='gray')
plt.title('Output train denoising')
plt.savefig('output_train_seg.png', dpi=500)

plt.imshow(X_train_noise[0].reshape(72,72), cmap='gray')
plt.title('Input train denoising')
plt.savefig('input_train_seg.png', dpi=500)

plt.imshow(Y_pred_without_noise[0].reshape(72,72), cmap='gray')
plt.title('Predicted Output test denoising')
plt.savefig('predict_output_test_seg.png', dpi=500)

plt.imshow(X_test_noise[0].reshape(72,72), cmap='gray')
plt.title('Input test denoising')
plt.savefig('input_test_seg.png', dpi=500)

