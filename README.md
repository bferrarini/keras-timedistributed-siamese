# keras-timedistributed-siamese

[tf.keras.layers.TimeDistributed](https://www.tensorflow.org/api_docs/python/tf/keras/layers/TimeDistributed) applies a sequence of samples to the same model and computes the loss accordingly.
It offers an alternative way to implement a multi-branch network (i.e. siamese).
This repository provides two scripts for both traditional and time-distributed implementation.
The two models are equivalent and expected to achieve comparable performance.

## Requisites
The code runs with: 

+ Tensorflow >= 2.0.0

## Run Siamese Experiment
```
python regular_siamese.py
```

## Run TimeDistributred Implementation Experiment
```
python timedistributed_siamese.py
```

## Performance
The MNIST is a lightweight model designed to be fast to train and achieve a state of the art performance. You can replace the corresponding section of the script with your network.
Thirty epochs should yield about 81% accuracy on the training set and 70% on the test set.
