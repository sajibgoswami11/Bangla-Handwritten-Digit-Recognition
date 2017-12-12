
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
import os
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

#x is array of 99 images of 784 flat array and y is just lebels of those 99
X = tf.placeholder(tf.float32, [data.shape[0],data.shape[1]])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y_ = tf.placeholder(tf.float32, [None, 10])


y=  tf.nn.softmax(tf.matmul(X,W)+b)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)


for i in range(1000):
    sess.run(train_step, feed_dict={X , y_ })
    if i % 100 == 0:
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("{0:3d} times\taccuracy: {1:.10f} %".format(i+100, sess.run(accuracy, feed_dict={X:data, y_: Label})*100))
    

