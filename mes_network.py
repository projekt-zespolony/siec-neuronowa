# input: timestamp, temperature, pressure, humidity, gas
# output: windows_opened, people_in_the_room

import xml_loader as xmll
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import backend as K
import os
import shutil


def main():

    (x_train, y_train), (x_test, y_test) = xmll.acquire_normalized_data('test_data.xml', 3 / 4)

    sess = tf.Session()
    K.set_session(sess)

    model = tf.keras.Sequential()

    model.add(layers.Dense(3, input_shape = x_train.shape[1:], activation = 'relu', name = 'input'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(3, activation = 'relu'))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(2, activation = 'sigmoid', name = 'output'))

    model.compile(optimizer = 'adam', 
        loss = 'binary_crossentropy',
        metrics = ['accuracy']
        )

    model.fit(x_train, y_train, epochs = 10, batch_size = 1, validation_data = (x_test, y_test))

    model.summary()

    shutil.rmtree("model", ignore_errors = True)
    builder = tf.saved_model.builder.SavedModelBuilder("model")
    builder.add_meta_graph_and_variables(sess, ["tag"])
    builder.save()

if __name__ == '__main__':
    main()
