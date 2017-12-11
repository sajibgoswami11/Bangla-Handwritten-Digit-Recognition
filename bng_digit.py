import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

img_rows , img_cols = 128 , 128

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


#%%
immatrix=array ([array(Image.open('C:\\Users\\User\\Desktop\\PycharmProjects\\images-resized' + '\\' + im2)).flatten() 
                    for im2 in imlist] ,'f')
       
m,n = immatrix.shape[0:2]

label = np.ones((num_samples,), dtype = int64 )
label[0:10]=0
label[11:40]=1
label[41:70]=2
label[71:90]=3
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
batch_size = 64
# number of output classes
nb_classes = 4
# number of epochs to train
nb_epoch = 20


# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

#%%
(X,y) =(train_data[0],train_data[1])

#split X ,y into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

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

i = 16
plt.imshow(X_train[i, 0], interpolation='nearest')
print("label : ", Y_train[i,:])




 #%%

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta')

#%%
#jodi input-shape[1,row,col] dei tahole ei vul dhore
#Negative dimension size caused by subtracting 3 from 1 for 'conv2d_1/convolution' (op: 'Conv2D') with input shapes: [?,1,128,128], [3,3,128,32].
#abar jodi [128,128,1] dei tahole 
#train 8

    model.compile(loss=keras.losses.categorical_crossentropy, \
                  optimizer=keras.optimizers.Adadelta(), \
                  metrics=['accuracy'])
#fit e ese milate pare na
#expected conv2d_3_input to have shape (None, 128, 128, 1) but got array with shape (19, 1, 128, 128)
    model.fit(X_train, Y_train, \
              batch_size=batch_size, \
              epochs=nb_epoch, \
              verbose=1, \
              validation_data=(X_test, Y_test))

    
    
    
    
    
