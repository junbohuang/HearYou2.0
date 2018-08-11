from tensorflow import keras
from metrics.top_k_accuracy import *


def load():


    input_layer = keras.layers.Input(shape=(200, 189, 1))
    layer = input_layer

    layer = keras.layers.Conv2D(32, 3, strides=(2,2), padding='same', activation='relu')(layer)
    layer = keras.layers.Dropout(0.2)(layer)
    layer = keras.layers.Conv2D(64, 3, strides=(2,2), padding='same', activation='relu')(layer)
    layer = keras.layers.Dropout(0.2)(layer)
    layer = keras.layers.Conv2D(64, 3, strides=(2,2), padding='same', activation='relu')(layer)
    layer = keras.layers.Dropout(0.2)(layer)
    layer = keras.layers.Conv2D(128, 3, strides=(2,2), padding='same', activation='relu')(layer)
    layer = keras.layers.Dropout(0.2)(layer)
    layer = keras.layers.Flatten()(layer)
    layer = keras.layers.Dense(1024, activation='relu')(layer)
    layer = keras.layers.Dropout(0.2)(layer)
    layer = keras.layers.Dense(256, activation='relu')(layer)
    layer = keras.layers.Dropout(0.2)(layer)

    output_layer = keras.layers.Dense(4, activation='softmax')(layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    metrics = top_k_accuracy()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)

    return model
