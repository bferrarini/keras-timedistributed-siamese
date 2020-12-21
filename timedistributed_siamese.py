"""
Copyright (c) 2020
Author:
  Bruno Ferrarini (University of Essex)
This project is licensed under MIT.

Timedistributed layer is used to implement a siamese network.

"""

import numpy as np
import random
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, TimeDistributed
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
import matplotlib.image as mpimg

from matplotlib import pyplot as plt

random.seed(10)

num_classes = 10
epochs = 30

show = False #displays the model diagram


def contrastive_loss(y_true, y_pred):
    q = y_pred[:,0,:]
    t = y_pred[:,1,:]
    sum_square = K.sum(K.square(q-t), axis=1, keepdims=True)
    d = K.sqrt(K.maximum(sum_square, K.epsilon()))
    
    margin = 1
    square_pred = K.square(d)
    margin_square = K.square(K.maximum(margin - d, 0))
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
    q = y_pred[:,0,:]
    t = y_pred[:,1,:]
    sum_square = K.sum(K.square(q-t), axis=1, keepdims=True)
    d = K.sqrt(K.maximum(sum_square, K.epsilon()))
    
    return K.mean(K.equal(y_true, K.cast(d < 0.5, y_true.dtype)))

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


time_input = Input(shape=time_input_shape, name = "TD_IN" )
time_out = TimeDistributed(base_network, name = "TD")(time_input)

model = Model(time_input, time_out)
model.summary()


if show:
    # you need graphviz
    plot_model(model, to_file="s-model.png", show_shapes=True, expand_nested=True)
    img = mpimg.imread('s-model.png')
    imgplot = plt.imshow(img)
    plt.show()    

rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
history = model.fit(tr_pairs, tr_y,
          batch_size=128,
          epochs=epochs,
          validation_data=(te_pairs, te_y))

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