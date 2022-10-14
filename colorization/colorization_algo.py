# import packages
import numpy  as np 
import matplotlib.pyplot as plt 
import cv2
import os

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model 
from tensorflow.keras import backend as k 
import tensorflowjs as tfjs

# IMPORT DATASETS

x_train=[]
x_test=[]
x_train_gray=[]
x_test_gray=[]

x_train_path=r"C:\Users\Charles lf\Desktop\custom_dataset\colorization\datasets\x_train"
x_test_path=r"C:\Users\Charles lf\Desktop\custom_dataset\colorization\datasets\x_test"

# the x_train dataset

for file_name in os.listdir(x_train_path):
    img1=cv2.imread(os.path.join(x_train_path, file_name))
    img1=cv2.resize(img1, (56,56))
    x_train.append(img1)

x_train=np.array(x_train)

#plt.imshow(x_train[0])
#plt.show()
print("x_train shape", x_train.shape)

# the x_test dataset

for file_name in os.listdir(x_test_path):
    img2=cv2.imread(os.path.join(x_test_path, file_name))
    img2=cv2.resize(img2, (56,56))
    x_test.append(img2)

x_test=np.array(x_test)
#plt.imshow(x_test[0])
#plt.show()
print("x_test shape", x_test.shape)

# the x_train_gray dataset

for file_name in os.listdir(x_train_path):
    img3=cv2.imread(os.path.join(x_train_path, file_name))
    img3=cv2.resize(img3, (56, 56))
    img3=cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    x_train_gray.append(img3)

x_train_gray=np.array(x_train_gray)

#plt.imshow(x_train_gray[0], cmap='gray')
#plt.show()
print("x_train_gray", x_train_gray.shape)

# the x_test_gray dataset

for file_name in os.listdir(x_test_path):
    img4=cv2.imread(os.path.join(x_test_path, file_name))
    img4=cv2.resize(img4, (56,56))
    img4=cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    x_test_gray.append(img4)

x_test_gray=np.array(x_test_gray)
#plt.imshow(x_test_gray[0],cmap='gray')
#plt.show()
print("x_test_gray shape ", x_test_gray.shape)

# IMAGE PREPROCESSING

# image size

img_rows= x_train.shape[1]
img_cols=x_train.shape[2]
channels=x_train.shape[3]

# normaize the output train and test color images
x_train=x_train.astype("float32")/ 255
x_test=x_test.astype('float32')/255

# normalize input train and test grayscale images
x_train_gray=x_train_gray.astype('float32')/255
x_test_gray=x_test_gray.astype('float32')/255

 # reshape images to row * col * channel for cnn output/validation

x_train=x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
x_test=x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)

# reshape image to row * col * channel for cnn input

x_train_gray= x_train_gray.reshape(x_train_gray.shape[0], img_rows, img_cols, 1)
x_test_gray=x_test_gray.reshape(x_test_gray.shape[0], img_rows, img_cols, 1)

# network parameters

input_shape=(img_rows, img_cols, 1)
batch_size=32
kernel_size=3
latent_dim=256

# encoder/decoder number of cnn layers and filters per layer

layers_filters=[64,128, 256]

# build the autoencoder model
# first build the encoder model

inputs=Input(shape=input_shape, name='encoder_input')

x=inputs

# stack of Conv2D

for filters in layers_filters:
    x=Conv2D(filters=filters, kernel_size=kernel_size, strides=2, activation='relu', padding='same')(x)

# shape info needed to build decoder model so we don't do hand computation
# the input to the decoder's first Conv2DTranspose will have this shape 

shape=k.int_shape(x)

# generate a latent vector

x=Flatten()(x)

latent=Dense(latent_dim, name='latent_vector')(x)

# instantiate encoder model

encoder = Model(inputs, latent, name='encoder')
encoder.summary()

# build the decoder model

latent_inputs= Input(shape=(latent_dim),name='decoder_input')
x=Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
x=Reshape((shape[1], shape[2], shape[3]))(x)

# stack of Conv2DTranspose

for filters in layers_filters[: : -1]:
    x=Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=2, activation='relu', padding='same')(x)

outputs=Conv2DTranspose(filters=channels,
                             kernel_size=kernel_size, activation='sigmoid', padding='same', name='decoder_output' )(x)

# instantiate decoder model

decoder=Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# instantiate autoencoder model

autoencoder=Model(inputs, decoder(encoder(inputs)), name='autoencoders')
autoencoder.summary()

# Mean square error (mse) loss function adam optimizer

autoencoder.compile(loss='mse', optimizer='adam')

# train the autoencoder

first_train=False
if first_train!=True:
    autoencoder.load_weights("model_weights/")
    print("weights loaded")
else:
    pass
    print("load wights... pass")

autoencoder.fit(x_train_gray, x_train, validation_data=(x_test_gray, x_test), epochs=20, batch_size=batch_size)

autoencoder.save_weights("model_weights/")

# serialize the model

#tfjs.converters.save_keras_model(autoencoder, "./models")

#autoencoder.save('keras_model_save/')

# TEST THE MODEL
x_decoded=autoencoder.predict(x_test_gray)

plt.imshow(x_test_gray[0], cmap='gray')
plt.show()

plt.imshow(x_decoded[0])
plt.show()

plt.imshow(x_test[0])
plt.show()