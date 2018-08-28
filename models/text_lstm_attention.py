from keras.layers import Input, Conv2D, Dropout, Flatten, concatenate
from keras.layers import BatchNormalization, Dense, Embedding, LSTM, Bidirectional
from keras.models import Model
from metrics.top_k_accuracy import *
from wrappers.attention import AttentionDecoder
from keras.optimizers import Adam


def load(nb_words, g_word_embedding_matrix, feat_size):

    input_layer = Input(shape=(500, ))
    layer = input_layer

    layer = Embedding(nb_words,
                                    300,
                                    weights=[g_word_embedding_matrix],
                                    input_length=500,
                                    trainable=True)(layer)
    layer = LSTM(256, return_sequences=True, recurrent_dropout=0.2)(layer)
    layer = Dropout(0.2)(layer)
    layer = LSTM(256, return_sequences=True, recurrent_dropout=0.2)(layer)
    layer = Dropout(0.2)(layer)
    layer = AttentionDecoder(256, 256)(layer)
    layer = Flatten()(layer)
    layer = Dense(256, activation='relu')(layer)
    #layer = BatchNormalization()(layer)

    output_layer = Dense(4, activation='softmax')(layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    metrics = top_k_accuracy()
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False, clipnorm=3.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=metrics)
    return model

