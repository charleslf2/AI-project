# import packages 

import numpy as np 
import cv2
import os
import matplotlib.pyplot as plt 

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as k
import tensorflowjs as tfjs

# load the input data

x_train_pixel=[]
x_test_pixel=[]
x_train=[]
x_test=[] 

x_train_pixel_path =r"C:\Users\Charles lf\Desktop\custom_dataset\pixale2highres\dataset\input_data\x_train_pixel"
x_test_pixel_path=r"C:\Users\Charles lf\Desktop\custom_dataset\pixale2highres\dataset\input_data\x_test_pixel"
x_train_path=r"C:\Users\Charles lf\Desktop\custom_dataset\pixale2highres\dataset\output_data\x_train"
x_test_path=r"C:\Users\Charles lf\Desktop\custom_dataset\pixale2highres\dataset\output_data\x_test"


# the x_train_pixe dataset
for file_name in os.listdir(x_train_pixel_path):
    img1=cv2.imread(os.path.join(x_train_pixel_path, file_name))
    img1=cv2.resize(img1, (200,200))
    x_train_pixel.append(img1)

x_train_pixel=np.array(x_train_pixel)

#plt.imshow(x_train_pixel[20])
#plt.show()
print("x_train_pixel shape", x_train_pixel.shape)


# the x_test_pixel datasets
for file_name in os.listdir(x_test_pixel_path):
    img2=cv2.imread(os.path.join(x_test_pixel_path, file_name))
    img2=cv2.resize(img2, (200,200))
    x_test_pixel.append(img2)

x_test_pixel=np.array(x_test_pixel)

#plt.imshow(x_test_pixel[20])
#plt.show()
print("x_test_pixel shape", x_test_pixel.shape)

#  the x_train datasets 

for file_name in os.listdir(x_train_path):
    img3=cv2.imread(os.path.join(x_train_path, file_name))
    img3=cv2.resize(img3, (200,200))
    x_train.append(img3)

x_train=np.array(x_train)

#plt.imshow(x_train[20])
#plt.show()
print("x_train shape", x_train.shape)

# the x_test datasets

for file_name in os.listdir(x_test_path):
    img4=cv2.imread(os.path.join(x_test_path, file_name))
    img4=cv2.resize(img4, (200,200))
    x_test.append(img4)

x_test=np.array(x_test)

#plt.imshow(x_test[20])
#plt.show()
print("x_test shape", x_test.shape)



# reshape to (700, 700, 3) and normalize input images
image_size=x_train.shape[1]

x_train=np.reshape(x_train, [-1, image_size, image_size, 3])
x_test=np.reshape(x_test, [-1, image_size, image_size, 3])

x_train_pixel=np.reshape(x_train_pixel, [-1, image_size, image_size, 3])
x_test_pixel=np.reshape(x_test_pixel, [-1, image_size, image_size, 3])

x_train=x_train.astype("float32")/250
x_test=x_test.astype("float32")/250

x_train_pixel=x_train_pixel.astype("float32")/250
x_test_pixel=x_test_pixel.astype("float32")/250

#plt.figure()
#plt.imshow(x_test_pixel[0])
#plt.colorbar()
#plt.show()
#plt.imshow(x_test[0])
#plt.show()

# network parameters

input_shape=(image_size, image_size, 3)
batch_size=32
kernel_size=3
latent_dim=256

# encoder/decoder number of CNN layers and filters per layer

layer_filters=[64, 128, 256]

# build the autoencoder model 
# first buold the encoder model

inputs=Input(shape=input_shape, name='encoder_input')

x=inputs

# stack of Conv2D(64)-Conv2D(125)-Conv2D(256)

for filters in layer_filters:
    x=Conv2D(filters=filters, kernel_size=kernel_size, 
    strides=2,activation='relu', padding='same')(x)

# shape info to build model show  no hand computation
# the input to the decoder's firts Conv2DTranspose will this shape

shape=k.int_shape(x)

print("the shape of k ", shape)

# generate the latent vector

x=Flatten()(x)
latent=Dense(latent_dim, name='latent_vector')(x)

# instanciate encoder model

encoder=Model(inputs, latent, name='encoder')
encoder.summary()

# build the decoder

latent_inputs=Input(shape=(latent_dim), name='decoder_input')

x=Dense(shape[1]*shape[2]*shape[3])(latent_inputs)

# from vector to suitable shape for transposed conv

x=Reshape((shape[1], shape[2], shape[3]))(x)

# the stack of Conv2DTranspose(256)-Conv2DTranspose(125)-Conv2DTranspose(64)

for filters in layer_filters[: :-1]:
    x=Conv2DTranspose(filters, kernel_size=kernel_size,
    strides=2, activation='relu', padding='same')(x)

# reconstruct the input

outputs=Conv2DTranspose(filters=1, kernel_size=kernel_size, 
padding='same', activation='sigmoid', name='decoder_output')(x)

# instantiate decoder model

decoder=Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# autoencoder = encoder + decoder

autoencoder=Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()

# Mean square error (MSE) loss function , Adam optimizer

autoencoder.compile(loss='mse', optimizer='adam')

# train the autoencoder

autoencoder.fit(x_train_pixel,x_train, 
validation_data=(x_test_pixel, x_test), epochs=5, batch_size=batch_size)

# save tensorflow js model
#tfjs.converters.save_keras_model(autoencoder, './models')

# save keras model
#autoencoder.save("keras_model_save/")

x_decoded=autoencoder.predict(x_test_pixel)


plt.imshow(x_test_pixel[20])
plt.show()

plt.imshow(x_decoded[20])
plt.show()