
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as k

import numpy as np 
import matplotlib.pyplot as plt 

# load datasets

(x_train,_), (x_test, _)=mnist.load_data()

# reshape to (28,28, 1) and normalize input images

image_size=x_train.shape[1]
x_train= np.reshape(x_train, [-1, image_size , image_size, 1])
x_test=np.reshape(x_test, [-1, image_size, image_size, 1])
x_train=x_train.astype('float32')/255
x_test = x_test.astype("float32")/255

# network parameters

input_shape=(image_size, image_size, 1)
batch_size=32
kernel_size=3
latent_dim=16

# encoder/decoder number of CNN layers and filters per layer

layer_filters=[32,64]

# build the autoencoder model
# first build  the encoder model

inputs=Input(shape=input_shape, name='encoder_input')
x=inputs
# stack to conv2D(32)-conv2D(64)

for filters in layer_filters:
    x=Conv2D(filters=filters, kernel_size=kernel_size,  
        activation='relu', strides=2, padding='same')(x)

# shape info needed to build decoder model
#so we don;t do hand computation
# the input to the decoder's first 
# Conv2DTranspose will have this shape
# shape is (7,7,64) wich processed by
# the decoder back to (28,28,1)

shape=k.int_shape(x)

# generate latent vector 

x=Flatten()(x)

latent= Dense(latent_dim, name='latent_vector')(x)

# instantiate encoder model

encoder=Model(inputs, latent, name='encoder')

encoder.summary()

# build the decoder model
latent_inputs=Input(shape=(latent_dim), name='decoder_input')

# use the shape (7,7,64) that was earlier saved

x=Dense(shape[1]* shape[2]*shape[3])(latent_inputs)

# from vector to suitable shape for transposed conv 
x=Reshape((shape[1], shape[2], shape[3]))(x)

# stack of Conv2DTranspose(64)- Conv2DTranspose(32)

for filters in layer_filters[: : -1]:
    x=Conv2DTranspose(filters=filters, kernel_size=kernel_size,
        activation='relu', strides=2, padding='same')(x)

# reconstruct the input 

outputs=Conv2DTranspose(filters=1, kernel_size=kernel_size,
activation='sigmoid', padding='same', name='decoder_output')(x)

# instantiate decoder model 

decoder =Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# autoencoder = encoder + decoder 
# instantiate autoencoder model 

autoencoder=Model(inputs, decoder(encoder(inputs))
, name='autoencoder')

autoencoder.summary()

# Mean squared error (MSE) loss function , Adam optimizer

autoencoder.compile(loss='mse', optimizer='adam')

# train the autoencoder 

autoencoder.fit(x_train, x_train, 
validation_data=(x_test, x_test), epochs=1, batch_size=batch_size)

# predict the autoencoder output from test data
x_decoder=autoencoder.predict(x_test)

# display the 1st 8 test input and decoded images

imgs=np.concatenate([x_test[:8], x_decoder[:8]])

imgs=np.vstack([np.hstack(i) for i in imgs])

plt.figure()

plt.axis('off')

plt.title('Input :1st 2 rows, Decoder : last 2rows')

plt.imshow(imgs, interpolation='none', cmap='gray')

plt.show()