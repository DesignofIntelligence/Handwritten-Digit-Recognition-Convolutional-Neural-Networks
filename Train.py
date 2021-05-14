import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
# Learning libraries
import warnings

warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


def load_data(directory):
    data = []  # Images will be stored here
    label = []  # Labels of images will be stored here
    for folder in os.listdir(directory + "trainingSet"):  # looping on the folders (from 0 to 9)
        for img in os.listdir("trainingSet/" + folder):  # looping on the images inside them.
            path = os.path.join(directory + "trainingSet/", folder, img)  # joining the full path of the image
            img_data = cv2.imread(path)  # reading this path using opencv
            img_data = cv2.resize(img_data, (28, 28))  # resizing the image to make sure its 28x28
            data.append(img_data)  # put the image into Data array
            x = int(folder)
            label.append(x)  # put its label in the folder array
    # shuffle both lists
    both = list(zip(data, label))  # join both arrays together
    random.shuffle(both)  # shuffle
    b: object
    a, b = zip(*both)  # disconnect them from each other

    return a, b  # return the data and label array.


def Train(images_array, label_array):
    # Change the arrays into numpy arrays, and reshaping them.
    X_train = np.array(images_array[:33600]).reshape(-1, 28, 28, 3)  # from 0 to 33600, reshaped to (33600,28,28,3)
    y_train = np.array(label_array[:33600])
    X_test = np.array(np.array(images_array[33600:])).reshape(8400, 28, 28,3)  # from 33600 to 42000, reshaped to (
    # 8400,28,28,3)
    y_test = np.array(label_array[33600:])

    # CREATING MODEL
    model = Sequential()  # make the model sequential
    # Convolutional layer, with 64 filters of 3x3 kernel size,
    # Then an activation function relu, then Max Pooling of size 2x2.
    model.add(Conv2D(64, (3, 3), input_shape=[28, 28, 3]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolutional layer, with 64 filters of 3x3 kernel size,
    # Then an activation function relu, then Max Pooling of size 2x2.
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolutional layer, with 64 filters of 3x3 kernel size,
    # Then an activation function relu, then Max Pooling of size 2x2.
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # Flatten the layer into a 1D input
    model.add(Dense(64))  # insert a neural network layer of 64 neurons
    model.add(Activation("relu"))  # Relu activation on them.

    model.add(Dense(32))  # insert a neural network layer of 32 neurons
    model.add(Activation("relu"))  # RELU activation on them

    model.add(Dense(10))  # insert OUTPUT layer of 10 neurons
    model.add(Activation("softmax"))  # Use softmax as activation function

    # compile this model, where our loss function will be sparse_categorical_crossentropy, optimizer adam,
    # and our metric will be accuracy
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train this model, 5 epochs on the training data, and 30% of this data to be used for validation.
    model.fit(X_train, y_train, epochs=5, validation_split=0.3)
    # evaluate this model based on new Data the model didnt see, (X_test), to calculate loss and accuracy.
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("loss is ", test_loss)
    print("accuracy is", test_acc)

    # predict on some training samples.
    predictions = model.predict([X_test])
    print("label is ", y_test[0])  # Actual label
    print("prediction is", np.argmax(predictions[0]))  # Prediction
    plt.imshow(X_test[0])   #plot image
    plt.show()

    print("label is ", y_test[1])
    print("prediction is", np.argmax(predictions[1]))
    plt.imshow(X_test[1])
    plt.show()

    # saving the model in tensorflow format
    model.save('my_model_weights.h5')


# Main
directory = "C:/Users/youss/Desktop/ASU semester 7/CSE 440 Selected Topics in Software Applications/HDR CNN project/"
images_array, label_array = load_data(directory)
Train(images_array, label_array)
