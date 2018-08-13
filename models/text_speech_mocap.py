from keras.layers import Input, Conv2D, Dropout, Flatten, concatenate
from keras.layers import BatchNormalization, Dense, Embedding, LSTM, Bidirectional
from keras.models import Model
from metrics.top_k_accuracy import *

def load(nb_words, g_word_embedding_matrix):


    text_input_layer = Input(shape=(500,))
    text_layer = text_input_layer
    text_layer = Embedding(nb_words,
                           300,
                           weights=[g_word_embedding_matrix],
                           input_length=500,
                           trainable=True)(text_layer)
    text_layer = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.2))(text_layer)
    text_layer = Dropout(0.2)(text_layer)
    text_layer = Bidirectional(LSTM(256, return_sequences=False, recurrent_dropout=0.2))(text_layer)
    text_layer = Dropout(0.2)(text_layer)
    text_layer = Dense(256, activation='relu')(text_layer)

    speech_input_layer = Input(shape=(100, 34))
    speech_layer = speech_input_layer

    speech_layer = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.2))(speech_layer)
    speech_layer = Dropout(0.2)(speech_layer)
    speech_layer = Bidirectional(LSTM(256, return_sequences=False, recurrent_dropout=0.2))(speech_layer)
    speech_layer = Dropout(0.2)(speech_layer)
    speech_layer = Dense(256, activation='relu')(speech_layer)

    mocap_input_layer = Input(shape=(200, 189, 1))
    mocap_layer = mocap_input_layer

    mocap_layer = Conv2D(32, 3, strides=(2, 2), padding='same', activation='relu')(mocap_layer)
    mocap_layer = BatchNormalization()(mocap_layer)
    mocap_layer = Dropout(0.2)(mocap_layer)
    mocap_layer = Conv2D(64, 3, strides=(2, 2), padding='same', activation='relu')(mocap_layer)
    mocap_layer = BatchNormalization()(mocap_layer)
    mocap_layer = Dropout(0.2)(mocap_layer)
    mocap_layer = Conv2D(64, 3, strides=(2, 2), padding='same', activation='relu')(mocap_layer)
    mocap_layer = BatchNormalization()(mocap_layer)
    mocap_layer = Dropout(0.2)(mocap_layer)
    mocap_layer = Conv2D(128, 3, strides=(2, 2), padding='same', activation='relu')(mocap_layer)
    mocap_layer = BatchNormalization()(mocap_layer)
    mocap_layer = Dropout(0.2)(mocap_layer)
    mocap_layer = Flatten()(mocap_layer)
    mocap_layer = Dense(256, activation='relu')(mocap_layer)

    combined_layer = concatenate([text_layer, speech_layer, mocap_layer])

    combined_layer = Dense(256, activation='relu')(combined_layer)
    output_layer = Dense(4, activation='softmax')(combined_layer)

    model = Model(inputs=[text_input_layer, speech_input_layer, mocap_input_layer], outputs=output_layer)

    metrics = top_k_accuracy()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)

    return model