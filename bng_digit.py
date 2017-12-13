from keras import optimizers
from keras.models import Sequential
from keras.layers import Activation,Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import Model
#from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from keras.optimizers import *
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
import os
import theano
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

img_rows , img_cols = 28 , 28

path1= 'C:\\Users\\User\\Desktop\\PycharmProjects\\images'
path2= 'C:\\Users\\User\\Desktop\\PycharmProjects\\images-resized'

listing = os.listdir(path1)

num_samples =size(listing)
print(num_samples)

for file in listing:
    im= Image.open(path1 + '\\' + file)
    img = im.resize ((img_rows,img_cols))
    gray = img.convert('L')
    
    gray.save(path2 + '\\' + file, "JPEG" )

imlist= os.listdir(path2)
im1 =array(Image.open('C:\\Users\\User\\Desktop\\PycharmProjects\\images-resized' + '\\' + imlist[0]))



immatrix=array ([array(Image.open('C:\\Users\\User\\Desktop\\PycharmProjects\\images-resized' + '\\' + im2)).flatten() 
                    for im2 in imlist] ,'f')
    
 #%%      
m,n = immatrix.shape[0:2]

label = np.ones((num_samples,), dtype = int64 )
label[0:8]=0
label[9:16]=1
label[17:27]=2
label[28:37]=3
label[38:47]=4
label[48:58]=5
label[59:67]=6
label[68:79]=7
label[80:87]=8
label[88:98]=9
print(label)  
    
  #%%
data,Label = shuffle(immatrix,label,random_state=2)
train_data = [data,Label]


#%%

plt.imshow(immatrix[16].reshape(img_rows,img_cols))
plt.imshow(immatrix[16].reshape(img_rows,img_cols),cmap='gray')
print (train_data[0].shape)
print (train_data[1].shape)

#parametres
#%%
#batch_size to train
batch_size = 32
# number of output classes
nb_classes = 10
# number of epochs to train
nb_epoch = 20


# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

#%%

(X, y) = (train_data[0],train_data[1])

plt.imshow(data[67].reshape(28,28))


#%%




#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print(X_train,y_train.shape)

#%%


#     X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#     X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#     input_shape=(1, img_rows, img_cols)



#%%
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols,1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols,1)
input_shape=( img_rows, img_cols,1)
 
   
  #%%     
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

i = 67
plt.imshow(X_train[i, 0], interpolation='nearest')
print("label : ", Y_train[i,:])



#%%
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',\
              optimizer='adadelta',\
              metrics=['accuracy'])
#%%
model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=20,
          verbose=1,
          validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])










