import tensorflow as tf
from tensorflow import keras
import numpy as np

def load_from_xml(xml_path):
    #not implemented yet

def main():

    (all_data, all_labels) = load_from_xml("Training_seq.xml") # Todo - loading from xml

    class_names = ["windowsOpened", "peopleInTheRoom"]

    # to modify
    tf_index = int(len(all_data) * (3.0 / 4.0))
    training_data = all_data[:tf_index]
    training_labels = all_labels[:tf_index]
    test_data = all_data[tf_index:]
    test_labels = all_labels[tf_index:]

    # if model already exists
    # tf.keras.models.load_model("saved_model/network_model")

    input_mes = 4
    input_neuron_num = input_mes * 4

    # if model don't exists
    model = keras.Sequential([
        keras.layers.Dense(input_neuron_num, activation = "relu"),
        keras.layers.Dense(24, activation = "relu"),
        keras.layers.Dense(2, activation = "softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])

    model.fit(training_data, training_labels, epochs = 10)

    predictions = model.predict(test_data)

    i = 0
    for p in predictions:
        if p == test_labels[i]:
            print(f"Test Sequence {i + 1}: Prediction success!")
        else:
            print(f"Test Sequence {i + 1}: Prediction fail!")
        i += 1

    model.save("saved_model/network_model")

if __name__ == "__main__":
    main()