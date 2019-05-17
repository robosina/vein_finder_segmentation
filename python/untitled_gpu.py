from __future__ import print_function
from model import *
from data import *
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import time as time
import cv2
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras import backend as K
import keras
from keras.models import Sequential
from keras import layers
from keras.layers import Dense,  Activation,Dropout
import tensorflow as tf
from keras.layers import Flatten
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization,concatenate,Input,UpSampling2D
x_shape=50 
from tensorflow.python.client import device_lib
from keras.preprocessing.image import ImageDataGenerator
print(device_lib.list_local_devices())
from keras import models
from keras.backend.tensorflow_backend import set_session  
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU  
sess = tf.Session(config=config)  
set_session(sess)  # set this TensorFlow session as the default session for Keras.

from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#%%

data_gen_args = dict(rotation_range=0.4,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
model = unet()
#model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
#model.fit_generator(myGene,steps_per_epoch=30,epochs=200)

#print('inputs: ', [input.op.name for input in model.inputs])
## outputs:  ['dense_4/Sigmoid']
#print('outputs: ', [output.op.name for output in model.outputs])
#model.save('./HODA.h5')
model.load_weights('/home/saeed/CUDA_IN_QT-master/first_layer/python/HODA.h5')
#
#frozen_graph = freeze_session(tf.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
#tf.train.write_graph(frozen_graph, './', 'ocr4.pbtxt', as_text=True)
#tf.train.write_graph(frozen_graph, './', 'ocr4.pb', as_text=False)

#history = model.fit_generator(
#      train_generator_final,
#      steps_per_epoch=40,
#      epochs=5)


#testGene = testGenerator("data/membrane/test")
#results = model.predict_generator(testGene,30,verbose=1)
#saveResult("data/membrane/test",results)
#%%
img=cv2.imread('/home/saeed/Music/11.jpg')
#plt.imshow(img,cmap='gray')
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img=cv2.resize(img,(256,256))
img=np.expand_dims(img,0)
img=np.expand_dims(img,3)
img=img/255.
t1=time.time()
out=model.predict(img)

out=np.squeeze(out)
fig=plt.figure(figsize=(4,4))
plt.imshow(out)
print(time.time()-t1)
out=np.abs(1-np.squeeze(out))
plt.figure()
fig=plt.figure(figsize=(8,8))
plt.imshow(out,cmap='gray')
fig=plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(np.squeeze(img),cmap='gray')
plt.subplot(1,2,2)
img=np.squeeze(img)
for row in range(0,256):
    for col in range(0,256):
        if(out[row,col]>0.01):
            img[row,col]=0+0.3*img[row,col]
            
# img=np.squeeze(img)-30*out
plt.imshow(img,cmap='gray')
cv2.imwrite('/home/nict/Desktop/1.jpg',img)
#%%    conv2d_1
from keras.models import Model
layer_name = 'conv2d_1'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
import cv2
data=cv2.imread('/home/saeed/Music/11.jpg')

data=cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
data=cv2.resize(data,(256,256))
data=np.array(data,dtype='float32')
data=data/255
data[0,0]=0.1
data[0,1]=0.2
data[0,2]=0.3
data[1,0]=0.4
data[1,1]=0.5
data[1,2]=0.6
data[2,0]=0.7
data[2,1]=0.8
data[2,2]=0.9
data[255,255]=0.1
data=np.expand_dims(data,0)
data=np.expand_dims(data,3)
intermediate_output = intermediate_layer_model.predict(data)
intermediate_output = np.squeeze(intermediate_output)
#
#d=np.squeeze(data)
data=np.squeeze(data)
np.savetxt('/home/saeed/CUDA_IN_QT-master/first_layer/weights/c1.txt',data,delimiter=',')
#np.savetxt('/home/nict/Documents/cuda_translate/c2.txt',d[:,:,1],delimiter=',')
#np.savetxt('/home/nict/Documents/cuda_translate/c3.txt',d[:,:,2],delimiter=',')
#
#inte=np.squeeze(intermediate_output)[:,:,0]
#np.savetxt('/home/nict/Documents/cuda_translate/in1.txt',inte,delimiter=',')
#
matrix=np.zeros((16,9),dtype='float32')
for i in range(0,16):
    w2=model.layers[1].get_weights()[0][:,:,:,i]
    w2=np.reshape(np.squeeze(w2),[1,9])
    matrix[i,:]=w2
np.savetxt('/home/saeed/CUDA_IN_QT-master/first_layer/python/conv2d_1_weights.txt',matrix,delimiter=',')

weigthts=np.squeeze(model.layers[1].get_weights()[0])
#matrix_bias=np.zeros((1,9),dtype='float32')
#for i in range(0,16):
#    bias=model.layers[1].get_weights()[1][:,:,:,i]
#    matrix_bias[i,:]=w2
matrix_bias=np.expand_dims(model.layers[1].get_weights()[1],0)
np.savetxt('/home/saeed/CUDA_IN_QT-master/first_layer/python/conv2d_1_bias.txt',matrix_bias,delimiter=',')
#np.savetxt('/home/nict/Documents/cuda_translate/w2.txt',w2[1,:,:],delimiter=',')
#np.savetxt('/home/nict/Documents/cuda_translate/w3.txt',w2[2,:,:],delimiter=',')
#
#%%   conv2d_2
layer_name = 'conv2d_2'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
import cv2
data=cv2.imread('/home/saeed/Music/11.jpg')
data=cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
data=cv2.resize(data,(256,256))
data=np.array(data,dtype='float32')
data=data/255
data[0,0]=0.1
data[0,1]=0.2
data[0,2]=0.3
data[1,0]=0.4
data[1,1]=0.5
data[1,2]=0.6
data[2,0]=0.7
data[2,1]=0.8
data[2,2]=0.9
data[255,255]=0.1
data=np.expand_dims(data,0)
data=np.expand_dims(data,3)
intermediate_conv2d_2 = intermediate_layer_model.predict(data)
intermediate_conv2d_2 = np.squeeze(intermediate_conv2d_2)  #get rid of batch size
w_conv2d_d=model.layers[2].get_weights()[0]
weight_matrix=np.zeros((16,144),dtype='float32')
for i in range(0,16):
    plane=w_conv2d_d[:,:,:,i]
    each_volume=np.empty( shape=(0, 0),dtype='float32' )
    for j in range(0,16):
        plane2=plane[:,:,j]
        p_reshape=np.expand_dims(np.squeeze(np.reshape(plane2,[1,9])),1)
        each_volume=np.append(each_volume,p_reshape)
    weight_matrix[i,:]=each_volume
np.savetxt('/home/saeed/CUDA_IN_QT-master/first_layer/python/conv2d_2_weights.txt',weight_matrix,delimiter=',')

conv2d_2_bias=np.expand_dims(model.layers[2].get_weights()[1],0)
np.savetxt('/home/saeed/CUDA_IN_QT-master/first_layer/python/conv2d_2_bias.txt',conv2d_2_bias,delimiter=',')

        
#%%
model.summary()       
layer_name = 'max_pooling2d_1'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
import cv2
data=cv2.imread('/home/saeed/Music/11.jpg')
data=cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
data=cv2.resize(data,(256,256))
data=np.array(data,dtype='float32')
data=data/255
data[0,0]=0.1
data[0,1]=0.2
data[0,2]=0.3
data[1,0]=0.4
data[1,1]=0.5
data[1,2]=0.6
data[2,0]=0.7
data[2,1]=0.8
data[2,2]=0.9
data[255,255]=0.1
data=np.expand_dims(data,0)
data=np.expand_dims(data,3)
intermediate_maxp2d_1 = intermediate_layer_model.predict(data)
intermediate_maxp2d_1 = np.squeeze(intermediate_maxp2d_1)  #get rid of batch size 
#%%
#%%
layer_name = 'conv2d_3'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
import cv2
data=cv2.imread('/home/saeed/Music/11.jpg')
data=cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
data=cv2.resize(data,(256,256))
data=np.array(data,dtype='float32')
data=data/255
data[0,0]=0.1
data[0,1]=0.2
data[0,2]=0.3
data[1,0]=0.4
data[1,1]=0.5
data[1,2]=0.6
data[2,0]=0.7
data[2,1]=0.8
data[2,2]=0.9
data[255,255]=0.1
data=np.expand_dims(data,0)
data=np.expand_dims(data,3)
intermediate_conv2d_3 = intermediate_layer_model.predict(data)
intermediate_conv2d_3 = np.squeeze(intermediate_conv2d_3)  #get rid of batch size
w_conv2d_3=model.layers[4].get_weights()[0]
weight_matrix_3=np.zeros((32,144),dtype='float32')
for i in range(0,32):
    plane=w_conv2d_3[:,:,:,i]
    each_volume=np.empty( shape=(0, 0),dtype='float32' )
    for j in range(0,16):
        plane2=plane[:,:,j]
        p_reshape=np.expand_dims(np.squeeze(np.reshape(plane2,[1,9])),1)
        each_volume=np.append(each_volume,p_reshape)
    weight_matrix_3[i,:]=each_volume
np.savetxt('/home/saeed/CUDA_IN_QT-master/first_layer/python/conv2d_3_weights.txt',weight_matrix_3,delimiter=',')

conv2d_3_bias=np.expand_dims(model.layers[4].get_weights()[1],0)
np.savetxt('/home/saeed/CUDA_IN_QT-master/first_layer/python/conv2d_3_bias.txt',conv2d_3_bias,delimiter=',')
w_conv2d_3_first_layer=np.squeeze(w_conv2d_3[:,:,:,0])
#%%
#%%
#%%
layer_name = 'conv2d_4'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
import cv2
data=cv2.imread('/home/saeed/Music/11.jpg')
data=cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
data=cv2.resize(data,(256,256))
data=np.array(data,dtype='float32')
data=data/255
data[0,0]=0.1
data[0,1]=0.2
data[0,2]=0.3
data[1,0]=0.4
data[1,1]=0.5
data[1,2]=0.6
data[2,0]=0.7
data[2,1]=0.8
data[2,2]=0.9
data[255,255]=0.1
data=np.expand_dims(data,0)
data=np.expand_dims(data,3)
intermediate_conv2d_4 = intermediate_layer_model.predict(data)
intermediate_conv2d_4 = np.squeeze(intermediate_conv2d_4)  #get rid of batch size
w_conv2d_4=model.layers[5].get_weights()[0]
weight_matrix_4=np.zeros((32,288),dtype='float32')
for i in range(0,32):
    plane=w_conv2d_4[:,:,:,i]
    each_volume=np.empty( shape=(0, 0),dtype='float32' )
    for j in range(0,32):
        plane2=plane[:,:,j]
        p_reshape=np.expand_dims(np.squeeze(np.reshape(plane2,[1,9])),1)
        each_volume=np.append(each_volume,p_reshape)
    weight_matrix_4[i,:]=each_volume
np.savetxt('/home/saeed/CUDA_IN_QT-master/first_layer/conv2d_4_weights.txt',weight_matrix_4,delimiter=',')

conv2d_4_bias=np.expand_dims(model.layers[5].get_weights()[1],0)
np.savetxt('/home/saeed/CUDA_IN_QT-master/first_layer/conv2d_4_bias.txt',conv2d_4_bias,delimiter=',')

w_conv2d_4_first_layer=np.squeeze(w_conv2d_4[:,:,:,0])        
w_conv2d_4_first_layer=np.squeeze(w_conv2d_4[:,:,:,1])        
#%%
model.summary()       
layer_name = 'max_pooling2d_2'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
import cv2
data=cv2.imread('/home/saeed/Music/11.jpg')
data=cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
data=cv2.resize(data,(256,256))
data=np.array(data,dtype='float32')
data=data/255
data[0,0]=0.1
data[0,1]=0.2
data[0,2]=0.3
data[1,0]=0.4
data[1,1]=0.5
data[1,2]=0.6
data[2,0]=0.7
data[2,1]=0.8
data[2,2]=0.9
data[255,255]=0.1
data=np.expand_dims(data,0)
data=np.expand_dims(data,3)
intermediate_maxp2d_2 = intermediate_layer_model.predict(data)
intermediate_maxp2d_2 = np.squeeze(intermediate_maxp2d_2)  #get rid of batch size 
#%%
layer_num=7
for i in range(0,5):
	disp(model.layers[layer_num].name)
nfilters=64   # change this
filter_depth=32
layer_name = 'conv2d_5'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
import cv2
data=cv2.imread('/home/saeed/Music/11.jpg')
data=cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
data=cv2.resize(data,(256,256))
data=np.array(data,dtype='float32')
data=data/255
data[0,0]=0.1
data[0,1]=0.2
data[0,2]=0.3
data[1,0]=0.4
data[1,1]=0.5
data[1,2]=0.6
data[2,0]=0.7
data[2,1]=0.8
data[2,2]=0.9
data[255,255]=0.1
data=np.expand_dims(data,0)
data=np.expand_dims(data,3)
intermediate_conv2d_5 = intermediate_layer_model.predict(data)
intermediate_conv2d_5 = np.squeeze(intermediate_conv2d_5)  #get rid of batch size
w_conv2d_5=model.layers[layer_num].get_weights()[0]
weight_matrix_5=np.zeros((nfilters,filter_depth*3*3),dtype='float32')
for i in range(0,nfilters):
    plane=w_conv2d_5[:,:,:,i]
    each_volume=np.empty( shape=(0, 0),dtype='float32' )
    for j in range(0,filter_depth):
        plane2=plane[:,:,j]
        p_reshape=np.expand_dims(np.squeeze(np.reshape(plane2,[1,9])),1)
        each_volume=np.append(each_volume,p_reshape)
    weight_matrix_5[i,:]=each_volume
np.savetxt('/home/saeed/CUDA_IN_QT-master/first_layer/weights/conv2d_5_weights.txt',weight_matrix_5,delimiter=',')

conv2d_5_bias=np.expand_dims(model.layers[layer_num].get_weights()[1],0)
np.savetxt('/home/saeed/CUDA_IN_QT-master/first_layer/weights/conv2d_5_bias.txt',conv2d_5_bias,delimiter=',')

w_conv2d_5_first_layer=np.squeeze(w_conv2d_5[:,:,:,0])        
w_conv2d_5_first_layer=np.squeeze(w_conv2d_5[:,:,:,1]) 
#%%
layer_num=8
for i in range(0,5):
	disp(model.layers[layer_num].name)
nfilters=64   # change this
filter_depth=64
layer_name = 'conv2d_6'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
import cv2
data=cv2.imread('/home/saeed/Music/11.jpg')
data=cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
data=cv2.resize(data,(256,256))
data=np.array(data,dtype='float32')
data=data/255
data[0,0]=0.1
data[0,1]=0.2
data[0,2]=0.3
data[1,0]=0.4
data[1,1]=0.5
data[1,2]=0.6
data[2,0]=0.7
data[2,1]=0.8
data[2,2]=0.9
data[255,255]=0.1
data=np.expand_dims(data,0)
data=np.expand_dims(data,3)
intermediate_conv2d_6 = intermediate_layer_model.predict(data)
intermediate_conv2d_6 = np.squeeze(intermediate_conv2d_6)  #get rid of batch size
w_conv2d_6=model.layers[layer_num].get_weights()[0]
weight_matrix_6=np.zeros((nfilters,filter_depth*3*3),dtype='float32')
for i in range(0,nfilters):
    plane=w_conv2d_6[:,:,:,i]
    each_volume=np.empty( shape=(0, 0),dtype='float32' )
    for j in range(0,filter_depth):
        plane2=plane[:,:,j]
        p_reshape=np.expand_dims(np.squeeze(np.reshape(plane2,[1,9])),1)
        each_volume=np.append(each_volume,p_reshape)
    weight_matrix_6[i,:]=each_volume
np.savetxt('/home/saeed/CUDA_IN_QT-master/first_layer/weights/conv2d_6_weights.txt',weight_matrix_6,delimiter=',')

conv2d_6_bias=np.expand_dims(model.layers[layer_num].get_weights()[1],0)
np.savetxt('/home/saeed/CUDA_IN_QT-master/first_layer/weights/conv2d_6_bias.txt',conv2d_6_bias,delimiter=',')

w_conv2d_6_first_layer=np.squeeze(w_conv2d_6[:,:,:,0])        
w_conv2d_6_first_layer=np.squeeze(w_conv2d_6[:,:,:,1])         
#%% max_pooling2d_3

layer_name = 'max_pooling2d_3'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
import cv2
data=cv2.imread('/home/saeed/Music/11.jpg')
data=cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
data=cv2.resize(data,(256,256))
data=np.array(data,dtype='float32')
data=data/255
data[0,0]=0.1
data[0,1]=0.2
data[0,2]=0.3
data[1,0]=0.4
data[1,1]=0.5
data[1,2]=0.6
data[2,0]=0.7
data[2,1]=0.8
data[2,2]=0.9
data[255,255]=0.1
data=np.expand_dims(data,0)
data=np.expand_dims(data,3)
intermediate_maxp2d_3 = intermediate_layer_model.predict(data)
intermediate_maxp2d_3 = np.squeeze(intermediate_maxp2d_3) 
#%% conv2d_7
layer_num=10
for i in range(0,5):
	disp(model.layers[layer_num].name)
nfilters=64   # change this
filter_depth=64
layer_name = model.layers[layer_num].name
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
import cv2
data=cv2.imread('/home/saeed/Music/11.jpg')
data=cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
data=cv2.resize(data,(256,256))
data=np.array(data,dtype='float32')
data=data/255
data[0,0]=0.1
data[0,1]=0.2
data[0,2]=0.3
data[1,0]=0.4
data[1,1]=0.5
data[1,2]=0.6
data[2,0]=0.7
data[2,1]=0.8
data[2,2]=0.9
data[255,255]=0.1
data=np.expand_dims(data,0)
data=np.expand_dims(data,3)
intermediate_conv2d_7 = intermediate_layer_model.predict(data)
intermediate_conv2d_7 = np.squeeze(intermediate_conv2d_7)  #get rid of batch size
w_conv2d_7=model.layers[layer_num].get_weights()[0]
weight_matrix_7=np.zeros((nfilters,filter_depth*3*3),dtype='float32')
for i in range(0,nfilters):
    plane=w_conv2d_7[:,:,:,i]
    each_volume=np.empty( shape=(0, 0),dtype='float32' )
    for j in range(0,filter_depth):
        plane2=plane[:,:,j]
        p_reshape=np.expand_dims(np.squeeze(np.reshape(plane2,[1,9])),1)
        each_volume=np.append(each_volume,p_reshape)
    weight_matrix_7[i,:]=each_volume
np.savetxt('/home/saeed/CUDA_IN_QT-master/first_layer/weights/conv2d_7_weights.txt',weight_matrix_7,delimiter=',')

conv2d_7_bias=np.expand_dims(model.layers[layer_num].get_weights()[1],0)
np.savetxt('/home/saeed/CUDA_IN_QT-master/first_layer/weights/conv2d_7_bias.txt',conv2d_7_bias,delimiter=',')

w_conv2d_7_first_layer=np.squeeze(w_conv2d_7[:,:,:,0])        
w_conv2d_7_first_layer=np.squeeze(w_conv2d_7[:,:,:,1])  
#%% conv2d_8
layer_num=11
for i in range(0,5):
	disp(model.layers[layer_num].name)
nfilters=64   # change this
filter_depth=64
layer_name = model.layers[layer_num].name
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
import cv2
data=cv2.imread('/home/saeed/Music/11.jpg')
data=cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
data=cv2.resize(data,(256,256))
data=np.array(data,dtype='float32')
data=data/255
data[0,0]=0.1
data[0,1]=0.2
data[0,2]=0.3
data[1,0]=0.4
data[1,1]=0.5
data[1,2]=0.6
data[2,0]=0.7
data[2,1]=0.8
data[2,2]=0.9
data[255,255]=0.1
data=np.expand_dims(data,0)
data=np.expand_dims(data,3)
intermediate_conv2d_8 = intermediate_layer_model.predict(data)
intermediate_conv2d_8 = np.squeeze(intermediate_conv2d_8)  #get rid of batch size
w_conv2d_8=model.layers[layer_num].get_weights()[0]
weight_matrix_8=np.zeros((nfilters,filter_depth*3*3),dtype='float32')
for i in range(0,nfilters):
    plane=w_conv2d_8[:,:,:,i]
    each_volume=np.empty( shape=(0, 0),dtype='float32' )
    for j in range(0,filter_depth):
        plane2=plane[:,:,j]
        p_reshape=np.expand_dims(np.squeeze(np.reshape(plane2,[1,9])),1)
        each_volume=np.append(each_volume,p_reshape)
    weight_matrix_8[i,:]=each_volume
np.savetxt('/home/saeed/CUDA_IN_QT-master/first_layer/weights/conv2d_8_weights.txt',weight_matrix_8,delimiter=',')

conv2d_8_bias=np.expand_dims(model.layers[layer_num].get_weights()[1],0)
np.savetxt('/home/saeed/CUDA_IN_QT-master/first_layer/weights/conv2d_8_bias.txt',conv2d_8_bias,delimiter=',')

w_conv2d_8_first_layer=np.squeeze(w_conv2d_8[:,:,:,0])        
w_conv2d_8_first_layer=np.squeeze(w_conv2d_8[:,:,:,1])  
#%% max_pooling2d_3

layer_name = 'max_pooling2d_4'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
import cv2
data=cv2.imread('/home/saeed/Music/11.jpg')
data=cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
data=cv2.resize(data,(256,256))
data=np.array(data,dtype='float32')
data=data/255
data[0,0]=0.1
data[0,1]=0.2
data[0,2]=0.3
data[1,0]=0.4
data[1,1]=0.5
data[1,2]=0.6
data[2,0]=0.7
data[2,1]=0.8
data[2,2]=0.9
data[255,255]=0.1
data=np.expand_dims(data,0)
data=np.expand_dims(data,3)
intermediate_maxp2d_4 = intermediate_layer_model.predict(data)
intermediate_maxp2d_4 = np.squeeze(intermediate_maxp2d_4)
#%% 
layer_num=14
for i in range(0,5):
	disp(model.layers[layer_num].name)
nfilters=128   # change this
filter_depth=64
layer_name = model.layers[layer_num].name
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
import cv2
data=cv2.imread('/home/saeed/Music/11.jpg')
data=cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
data=cv2.resize(data,(256,256))
data=np.array(data,dtype='float32')
data=data/255
data[0,0]=0.1
data[0,1]=0.2
data[0,2]=0.3
data[1,0]=0.4
data[1,1]=0.5
data[1,2]=0.6
data[2,0]=0.7
data[2,1]=0.8
data[2,2]=0.9
data[255,255]=0.1
data=np.expand_dims(data,0)
data=np.expand_dims(data,3)
intermediate_conv2d_9 = intermediate_layer_model.predict(data)
intermediate_conv2d_9 = np.squeeze(intermediate_conv2d_9)  #get rid of batch size
w_conv2d_9=model.layers[layer_num].get_weights()[0]
weight_matrix_9=np.zeros((nfilters,filter_depth*3*3),dtype='float32')
for i in range(0,nfilters):
    plane=w_conv2d_9[:,:,:,i]
    each_volume=np.empty( shape=(0, 0),dtype='float32' )
    for j in range(0,filter_depth):
        plane2=plane[:,:,j]
        p_reshape=np.expand_dims(np.squeeze(np.reshape(plane2,[1,9])),1)
        each_volume=np.append(each_volume,p_reshape)
    weight_matrix_9[i,:]=each_volume
np.savetxt('/home/saeed/CUDA_IN_QT-master/first_layer/weights/conv2d_9_weights.txt',weight_matrix_9,delimiter=',')

conv2d_9_bias=np.expand_dims(model.layers[layer_num].get_weights()[1],0)
np.savetxt('/home/saeed/CUDA_IN_QT-master/first_layer/weights/conv2d_9_bias.txt',conv2d_9_bias,delimiter=',')

w_conv2d_9_first_layer=np.squeeze(w_conv2d_9[:,:,:,0])        
w_conv2d_9_first_layer=np.squeeze(w_conv2d_9[:,:,:,1])  
#%% 
layer_num=15
for i in range(0,5):
	disp(model.layers[layer_num].name)
nfilters=128   # change this
filter_depth=128
layer_name = model.layers[layer_num].name
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
import cv2
data=cv2.imread('/home/saeed/Music/11.jpg')
data=cv2.cvtColor(data,cv2.COLOR_BGR2GRAY)
data=cv2.resize(data,(256,256))
data=np.array(data,dtype='float32')
data=data/255
data[0,0]=0.1
data[0,1]=0.2
data[0,2]=0.3
data[1,0]=0.4
data[1,1]=0.5
data[1,2]=0.6
data[2,0]=0.7
data[2,1]=0.8
data[2,2]=0.9
data[255,255]=0.1
data=np.expand_dims(data,0)
data=np.expand_dims(data,3)
intermediate_conv2d_10 = intermediate_layer_model.predict(data)
intermediate_conv2d_10 = np.squeeze(intermediate_conv2d_10)  #get rid of batch size
w_conv2d_10=model.layers[layer_num].get_weights()[0]
weight_matrix_10=np.zeros((nfilters,filter_depth*3*3),dtype='float32')
for i in range(0,nfilters):
    plane=w_conv2d_10[:,:,:,i]
    each_volume=np.empty( shape=(0, 0),dtype='float32' )
    for j in range(0,filter_depth):
        plane2=plane[:,:,j]
        p_reshape=np.expand_dims(np.squeeze(np.reshape(plane2,[1,9])),1)
        each_volume=np.append(each_volume,p_reshape)
    weight_matrix_10[i,:]=each_volume
np.savetxt('/home/saeed/CUDA_IN_QT-master/first_layer/weights/conv2d_10_weights.txt',weight_matrix_10,delimiter=',')

conv2d_10_bias=np.expand_dims(model.layers[layer_num].get_weights()[1],0)
np.savetxt('/home/saeed/CUDA_IN_QT-master/first_layer/weights/conv2d_10_bias.txt',conv2d_10_bias,delimiter=',')

w_conv2d_10_first_layer=np.squeeze(w_conv2d_10[:,:,:,0])        
w_conv2d_10_first_layer=np.squeeze(w_conv2d_10[:,:,:,0])  