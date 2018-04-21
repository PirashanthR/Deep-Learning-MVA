# -*- coding: utf-8 -*-
"""
This file contains what I have implemented for the first TP of the deep learning class
in order to solve the regression task
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

############5 A Regression Problem#############

#Create the datasets
[X_train, Y_train] = generate_dataset_regression(5000, 20)
[X_test, Y_test] = generate_test_set_regression()

#Normalize the data using sklearn tool
normalizer = StandardScaler()
y_train_normalize = normalizer.fit_transform(Y_train)
y_test_normalize = normalizer.transform(Y_test)


#visualize_prediction(X_train[0], Y_train[0])
nb_of_features_output = 6

#A basic linear regressor model
model =Sequential()
model.add(Dense(nb_of_features_output,input_shape=(X_train.shape[1],)))


#Compile the model 
model.compile(loss='mean_squared_error',
              optimizer='adam')

history= model.fit(X_train,y_train_normalize,validation_data=(X_test,y_test_normalize),epochs=100,batch_size=32)


#A Deep Neural Network with the sequential model
#This model actually performs the best

#[X_train, Y_train] = generate_dataset_regression(2000, 20)
#[X_test, Y_test] = generate_test_set_regression()


X_train = X_train.reshape((-1,72,72,1))
X_test = X_test.reshape((-1,72,72,1))
normalizer = StandardScaler()
y_train_normalize = normalizer.fit_transform(Y_train)
y_test_normalize = normalizer.transform(Y_test)
nb_of_features_output = 6

mcp = ModelCheckpoint('weights.best_reg.hdf5', monitor="val_loss",
                      save_best_only=True, save_weights_only=False)
model =Sequential()

model.add(Convolution2D(40,kernel_size=(5,5),input_shape=(72,72,1,),activation='relu'))
model.add(Dropout(0.3))
model.add(Convolution2D(20,kernel_size=(4,4),activation='relu'))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(nb_of_features_output))


#Compile the model 
model.compile(loss='mean_squared_error',
              optimizer='adam')

history= model.fit(X_train,y_train_normalize,validation_split=0.1,epochs=100,batch_size=32,callbacks=[mcp])

model = keras.models.load_model('weights.best_reg.hdf5')

predictions_test = model.predict(X_test)

Y_pred = normalizer.inverse_transform(predictions_test)
visualize_prediction(X_test[100], Y_pred[100])


print('Regularized network score ',model.evaluate(X_test, y_test_normalize))

#A more complex model with the API model

from keras.models import Model
#from keras.layers import 
from keras.layers import Input, Embedding, Dropout, Conv2D , Dense,BatchNormalization

mcp = ModelCheckpoint('weights.best_reg1.hdf5', monitor="val_loss",
                      save_best_only=True, save_weights_only=False)

input_data = Input(shape = (72,72,1,))

layer_1 = Conv2D(40,kernel_size=(2,2),activation='relu')(input_data)
layer_2 = Flatten()(layer_1)
layer_3 = Dropout(0.3)(layer_2)
layer_4 = Dense(50,activation='relu')(layer_3)
layer_5 = (layer_4)

####Prediction for different outputs

layer_1_bis = Dropout(0.3) Dense(40,activation='relu')(layer_5)
output_1 = Dense(1)(layer_1_bis)

layer_2_bis = Dense(20,activation='relu')(layer_5)
output_2 = Dense(1)(layer_2_bis)

layer_3_bis = Dense(20,activation='relu')(layer_5)
output_3 = Dense(1)(layer_3_bis)

layer_4_bis = Dense(20,activation='relu')(layer_5)
output_4 = Dense(1)(layer_4_bis)

layer_5_bis = Dense(20,activation='relu')(layer_5)
output_5 = Dense(1)(layer_5_bis)

layer_6_bis = Dense(20,activation='relu')(layer_5)
output_6= Dense(1)(layer_6_bis)

final_output = keras.layers.concatenate([output_1,output_2,output_3,\
                                         output_4,output_5,output_6],axis=1)

task_separ_model = Model(input_data,final_output)
task_separ_model.compile(loss='mean_squared_error',optimizer='adam')


history= task_separ_model.fit(X_train,y_train_normalize,validation_split=0.1,epochs=50,batch_size=32,callbacks=[mcp])

model = keras.models.load_model('weights.best_reg1.hdf5')


print('Regularized API network score ',model.evaluate(X_test, y_test_normalize))

