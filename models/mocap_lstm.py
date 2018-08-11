from tensorflow import keras
from metrics.top_k_accuracy import *


def load():


    input_layer = keras.layers.Input(shape=(200, 189))
    layer = input_layer

    layer = keras.layers.LSTM(256, return_sequences=True, recurrent_dropout=0.2)(layer)
    layer = keras.layers.Dropout(0.2)(layer)
    layer = keras.layers.LSTM(256, return_sequences=False, recurrent_dropout=0.2)(layer)
    layer = keras.layers.Dropout(0.2)(layer)
    layer = keras.layers.Dense(256, activation='relu')(layer)

    output_layer = keras.layers.Dense(4, activation='softmax')(layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    metrics = top_k_accuracy()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)

    return model