from keras.layers import Input, Conv2D, Conv3D, Dropout, Flatten, Reshape, MaxPooling2D
from keras.layers import BatchNormalization, Dense, Embedding, LSTM, Bidirectional
from keras.models import Model
from metrics.top_k_accuracy import *
from wrappers.attention import AttentionDecoder
from tensorflow.python.keras import backend as K



def load():


    input_layer = Input(shape=(100, 34, 3))
    layer = input_layer

    layer = Conv2D(128, kernel_size=(5, 3), strides=(1, 1), padding='same', activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D()(layer)
    layer = Dropout(0.2)(layer)
    layer = Conv2D(256, kernel_size=(5, 3), strides=(1, 1), padding='same', activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.2)(layer)
    layer = Conv2D(256, kernel_size=(5, 3), strides=(1, 1), padding='same', activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.2)(layer)
    layer = Reshape((-1, 100))(layer)

    layer = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2))(layer)
    layer = Dropout(0.2)(layer)
    layer = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2))(layer)
    layer = Dropout(0.2)(layer)
    layer = AttentionDecoder(256, 256)(layer)
    layer = Flatten()(layer)


    output_layer = Dense(4, activation='softmax')(layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    metrics = top_k_accuracy()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)

    return model