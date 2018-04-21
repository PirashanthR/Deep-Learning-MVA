# -*- coding: utf-8 -*-
"""
The file contains what I have implemented for the first part of the TP that deals
with classification.
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

def visualize_and_print_loss(history,name_for_plot_loss,name_for_plot_acc):
    '''
    Tool to plot the loss and the accuracy trough epochs
    @param: history -> keras history object that contains the needed data
    '''
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('name_for_plot_acc.png', dpi=500)
    
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('name_for_plot_loss.png', dpi=500)
    plt.show()

[X_train, Y_train] = generate_dataset_classification(300, 20)
[X_test, Y_test] = generate_test_set_classification()



############2 Simple Classification#############

Y_train_categorical = keras.utils.to_categorical(Y_train)
#Parameters
nb_of_classes = 3

#Create linear model
model =Sequential()

model.add(Dense(nb_of_classes,input_shape=(X_train.shape[1],),activation = 'softmax'))

#Compile the mode 
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics=['acc']) 

#Fit the model
history= model.fit(X_train,Y_train_categorical,validation_data=(X_test,Y_test),epochs=10,batch_size=32)

################################################

############3 Visualization of the Solution#############
W,b = model.get_weights()
plt.figure(1)
plt.imshow(W[:,0].reshape(72,72), cmap='gray')
plt.title('1rst column - rectangle')
plt.savefig('1rstcolumn.png', dpi=500)


plt.figure(2)
plt.imshow(W[:,1].reshape(72,72), cmap='gray')
plt.title('2nd column - circle')
plt.savefig('2ndcolumn.png', dpi=500)

plt.figure(3)
plt.imshow(W[:,2].reshape(72,72), cmap='gray')
plt.title('3rd column - triangle')
plt.savefig('3rdcolumn.png', dpi=500)
################################################

############4 A More Difficult Classification Problem#############
[X_train, Y_train] = generate_dataset_classification(10000, 20, True) #Improve nb of data to improve the score
[X_test, Y_test] = generate_test_set_classification()

Y_train_categorical = keras.utils.to_categorical(Y_train)

X_train = X_train.reshape((-1,72,72,1))
X_test = X_test.reshape((-1,72,72,1))

#Architecture described in the TP
model =Sequential()

model.add(Convolution2D(16,kernel_size=(5,5),input_shape=(72,72,1,)))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Flatten())
model.add(Dense(nb_of_classes,activation = 'softmax'))

###########Compile the model################### 
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics=['acc']) 

history= model.fit(X_train,Y_train_categorical,validation_data=(X_test,Y_test),epochs=150,batch_size=32)

print('Model described in class score: ',model.evaluate(X_test, Y_test))
################################################

#Try Using VGG for the classification task
from keras.applications import VGG16
new_model = VGG16(input_shape=(72,72,1),classes=3,weights=None)
new_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics=['acc']) 
hist = new_model.fit(X_train,Y_train_categorical,validation_data=(X_test,Y_test),epochs=20,batch_size=32)

#Regularize the architecture seen in TP to improve the val score
#Model checkpoint 
mcp = ModelCheckpoint('weights.best.hdf5', monitor="val_acc",
                      save_best_only=True, save_weights_only=False)

model =Sequential()

model.add(Convolution2D(40,kernel_size=(5,5),input_shape=(72,72,1,)))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(nb_of_classes,activation = 'softmax'))

###########Compile the model################### 
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics=['acc']) 

history= model.fit(X_train,Y_train_categorical,validation_split=0.1,epochs=150,batch_size=32,callbacks=[mcp])

model = keras.models.load_model('weights.best.hdf5')

print('Regularized network score: ',model.evaluate(X_test, Y_test))

#I finally obtained about 95% accuracy with this last model



