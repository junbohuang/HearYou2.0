from tensorflow import keras
from metrics.top_k_accuracy import *


def load(nb_words, g_word_embedding_matrix):

    text_input_layer = keras.layers.Input(shape=(500, ))
    text_layer = text_input_layer
    text_layer = keras.layers.Embedding(nb_words,
                                   300,
                                   weights=[g_word_embedding_matrix],
                                   input_length=500,
                                   trainable=True)(text_layer)
    text_layer = keras.layers.LSTM(256, return_sequences=True, recurrent_dropout=0.2)(text_layer)
    text_layer = keras.layers.Dropout(0.2)(text_layer)
    text_layer = keras.layers.LSTM(256, return_sequences=False, recurrent_dropout=0.2)(text_layer)
    text_layer = keras.layers.Dropout(0.2)(text_layer)
    text_layer = keras.layers.Dense(256, activation='relu')(text_layer)

    speech_input_layer = keras.layers.Input(shape=(100, 34))
    speech_layer = speech_input_layer

    speech_layer = keras.layers.LSTM(256, return_sequences=True, recurrent_dropout=0.2)(speech_layer)
    speech_layer = keras.layers.Dropout(0.2)(speech_layer)
    speech_layer = keras.layers.LSTM(256, return_sequences=False, recurrent_dropout=0.2)(speech_layer)
    speech_layer = keras.layers.Dropout(0.2)(speech_layer)
    speech_layer = keras.layers.Dense(256, activation='relu')(speech_layer)

    combined_layer = keras.layers.concatenate([text_layer, speech_layer])

    output_layer = keras.layers.Dense(4, activation='softmax')(combined_layer)

    model = keras.models.Model(inputs=[text_input_layer, speech_input_layer], outputs=output_layer)

    metrics = top_k_accuracy()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)

    return model