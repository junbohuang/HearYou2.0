from tensorflow import keras
from metrics.top_k_accuracy import *


def load():

    input_layer = keras.layers.Input(shape=(100, 34))
    layer = input_layer

    layer = keras.layers.Flatten(input_shape=(100, 34))(layer)
    layer = keras.layers.Dense(1024, activation='relu')(layer)
    layer = keras.layers.Dropout(0.2)(layer)
    layer = keras.layers.Dense(512, activation='relu')(layer)
    layer = keras.layers.Dropout(0.2)(layer)
    layer = keras.layers.Dense(256, activation='relu')(layer)
    layer = keras.layers.Dropout(0.2)(layer)

    output_layer = keras.layers.Dense(4, activation='softmax')(layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    metrics = top_k_accuracy()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)

    return model

