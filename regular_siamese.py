"""
Copyright (c) 2020
Author:
  Bruno Ferrarini (University of Essex)
This project is licensed under MIT.

The source code is a modified version of the example shared at https://github.com/aspamers/siamese/blob/master/mnist_siamese_example.py to work with Tensorflow 2.0
A siamese network is implemented with the regular twin-branch architecture and trained on MNIST.
The performance metric is the accuracy in matching digits of the same class.

"""

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Lambda, Conv2D
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.backend as K

from matplotlib import pyplot as plt

import random

# this should make the process repeatable
random.seed(10)

num_classes = 10
epochs = 30


def euclidean_distance(vects):
    x,y = vects
    sum_square = K.sum(K.square(x-y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1-y_true) * margin_square)

def create_pairs(x, digit_indices):
    
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]] # positive sample
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i] # negative sample
            pairs += [[x[z1], x[z2]]]
            labels += [1.0, 0.0]
    return np.array(pairs), np.array(labels)


def create_base_model(input_shape):
    model_input = Input(shape=input_shape)

    embedding = Conv2D(32, kernel_size=(3, 3), input_shape=input_shape)(model_input)
    embedding = BatchNormalization()(embedding)
    embedding = Activation(activation='relu')(embedding)
    embedding = MaxPooling2D(pool_size=(2, 2))(embedding)
    embedding = Conv2D(64, kernel_size=(3, 3))(embedding)
    embedding = BatchNormalization()(embedding)
    embedding = Activation(activation='relu')(embedding)
    embedding = MaxPooling2D(pool_size=(2, 2))(embedding)
    embedding = Flatten()(embedding)
    embedding = Dense(128)(embedding)
    embedding = BatchNormalization()(embedding)
    embedding = Activation(activation='relu')(embedding)

    return Model(model_input, embedding)

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


(x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
input_shape = x_train.shape[1:]
print("Base model input shape: {}".format(input_shape))
time_input_shape = (2, input_shape[0], input_shape[1], input_shape[2])
print("T model input shape: {}".format(time_input_shape))

digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)

digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
te_pairs, te_y = create_pairs(x_test, digit_indices)


print("Shape of training pairs", tr_pairs.shape)
print("Shape of training labels", tr_y.shape)


# network definition
base_network = create_base_model(input_shape)
base_network.summary()

input_a = Input(shape=input_shape, name='input_a')
input_b = Input(shape=input_shape, name='input_b')

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)


rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])

history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=128,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y)
          )

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()