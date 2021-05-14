import tensorflow as tf
import cv2
import os
import numpy as np

from keras.models import load_model
import matplotlib.pyplot as plt

import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


def Predict(image_path):
    data = []  # Images will be stored here
    label = []  # Labels will be stored here
    for img in os.listdir(image_path):  # Path to be explored
        path = os.path.join(image_path, img)  # joining the full path of the image
        img_data = cv2.imread(path)  # Read the image
        img_data = cv2.resize(img_data, (28, 28))  # Make sure its 28x28
        data.append(img_data)  # Put image into data array

    x = np.array(data)  # change it into numpy array

    new_model = load_model('my_model_weights.h5')  # load the model
    plt.imshow(x[0])
    plt.show()  # plot the image
    predictions = new_model.predict([x])
    print("prediction is", np.argmax(predictions[0]))  # Predict the image label
    return np.argmax(predictions[0])


# ● Print and return the predicted label for this image

Predict('MyOwnImages/')
