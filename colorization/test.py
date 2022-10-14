import numpy as np
import matplotlib.pyplot as plt 
import cv2 
import os 

from tensorflow import keras

test=[]
test_path=r"C:\Users\Charles lf\Desktop\custom_dataset\colorization\datasets\test"

for file_name in os.listdir(test_path):
    img=cv2.imread(os.path.join(test_path, file_name))
    img=cv2.resize(img, (32,32))
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    test.append(img)

test=np.array(test)

print("test shape ", test.shape)

#plt.imshow(test[0], cmap='gray')
#plt.show()


# reshape the test image
test=test.reshape(test.shape[0], test.shape[1], test.shape[2], 1)

# load the model

colorization=keras.models.load_model("keras_model_save/")

# make prediction

result=colorization.predict(test)

plt.imshow(test[2], cmap='gray')
plt.show()

plt.imshow(result[2])
plt.show()



