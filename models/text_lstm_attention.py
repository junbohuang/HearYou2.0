from keras.layers import Input, Conv2D, Dropout, Flatten, concatenate
from keras.layers import BatchNormalization, Dense, Embedding, LSTM, Bidirectional
from keras.models import Model
from metrics.top_k_accuracy import *
from wrappers.attention import AttentionDecoder

def load(nb_words, g_word_embedding_matrix):

    input_layer = Input(shape=(500, ))
    layer = input_layer

    layer = Embedding(nb_words,
                                    300,
                                    weights=[g_word_embedding_matrix],
                                    input_length=500,
                                    trainable=True)(layer)
    layer = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.2))(layer)
    layer = Dropout(0.2)(layer)
    layer = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.2))(layer)
    layer = Dropout(0.2)(layer)
    layer = AttentionDecoder(256, 256)(layer)
    layer = Flatten()(layer)
    layer = Dense(256, activation='relu')(layer)

    output_layer = Dense(4, activation='softmax')(layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    metrics = top_k_accuracy()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)
    return model

