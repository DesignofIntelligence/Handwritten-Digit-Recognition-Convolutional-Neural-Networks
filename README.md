# Handwritten-Digit-Recognition-Convolutional-Neural-Networks
A Convolutional Neural Network that is able to recognize handwritten digits. trained network (.h5 file) and training data are included.

Model Summary:
1st Layer: Convolutional layer, with 64 filters of 3x3 kernel size, Then an 
activation function relu, then Max Pooling of size 2x2.

2nd Layer: Convolutional layer, with 64 filters of 3x3 kernel size, Then an 
activation function relu, then Max Pooling of size 2x2.

3rd Layer: Convolutional Layer, with 64 filters of 3x3 kernel size, Then an 
activation function relu, then Max Pooling of size 2x2.

4th Layer: Neural network dense layer (also hidden layer) with 64 
neurons and activation function relu.

5th Layer: Neural network dense layer (also hidden layer) with 32 
neurons and activation function relu.

6th Layer: Output Layer of 10 outputs with softmax activation function.
Input Size is 28x28 RGB, so 28x28x3

Output size: 10 values ranging from 0 to 9
Training data = 33600 (80%)
Testing Data = 8400 (20%)
Total data size = 42000

HyperParameters:
Number of epochs: 5
Activation function: relu in all except output, softmax in output.
Number of hidden layers: 2, with 64 neuron in the first and 32 in the second

Enviroment:
TensorFlow version: 2.4
Keras Version: 2.4.3
NumPy Version: 1.19.5
OpenCV Version: 4.5.1.48
Matplotlib: 3.3.3
