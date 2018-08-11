from tensorflow import keras
from metrics.top_k_accuracy import *


def load(nb_words, g_word_embedding_matrix):

    input_layer = keras.layers.Input(shape=(500, ))
    layer = input_layer

    layer = keras.layers.Embedding(nb_words,
                                    300,
                                    weights=[g_word_embedding_matrix],
                                    input_length=500,
                                    trainable=True)(layer)
    layer = keras.layers.LSTM(256, return_sequences=True, recurrent_dropout=0.2)(layer)
    layer = keras.layers.LSTM(256, return_sequences=False, recurrent_dropout=0.2)(layer)
    layer = keras.layers.Dense(256)(layer)

    output_layer = keras.layers.Dense(4, activation='softmax')(layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    metrics = top_k_accuracy()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)
    return model

