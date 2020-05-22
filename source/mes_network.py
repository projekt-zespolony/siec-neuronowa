import xml_loader as xmll
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import os


def main():

    (x_train, y_train), (x_test, y_test) = xmll.acquire_normalized_data('test_data.xml', 3 / 4)

    print(len(x_train))

    model = tf.keras.Sequential()

    model.add(layers.Dense(3, input_shape = x_train.shape[1:], activation = 'relu'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(3, activation = 'relu'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(2, activation = 'sigmoid'))

    model.compile(optimizer = 'adam', 
        loss = 'binary_crossentropy',
        metrics = ['accuracy']
        )

    model.fit(x_train, y_train, epochs = 10, batch_size = 1, validation_data = (x_test, y_test))

if __name__ == '__main__':
    main()
