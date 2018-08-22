from keras.layers import Input, Conv2D, Conv3D, Dropout, Flatten, Reshape, MaxPooling2D, concatenate
from keras.layers import BatchNormalization, Dense, Embedding, LSTM, Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from metrics.top_k_accuracy import *
from wrappers.attention import AttentionDecoder
from tensorflow.python.keras import backend as K

def load():

    speech_input_layer = Input(shape=(100, 12, 3))
    speech_layer = speech_input_layer

    speech_layer = Conv2D(128, kernel_size=(5, 3), strides=(1, 1), padding='same', activation='relu')(speech_layer)
    speech_layer = MaxPooling2D(padding='same')(speech_layer)
    speech_layer = Dropout(0.2)(speech_layer)
    speech_layer = Conv2D(256, kernel_size=(5, 3), strides=(1, 1), padding='same', activation='relu')(speech_layer)
    speech_layer = Dropout(0.2)(speech_layer)
    speech_layer = Conv2D(256, kernel_size=(5, 3), strides=(1, 1), padding='same', activation='relu')(speech_layer)
    speech_layer = Dropout(0.2)(speech_layer)
    speech_layer = Conv2D(256, kernel_size=(5, 3), strides=(1, 1), padding='same', activation='relu')(speech_layer)
    speech_layer = Dropout(0.2)(speech_layer)
    speech_layer = Conv2D(256, kernel_size=(5, 3), strides=(1, 1), padding='same', activation='relu')(speech_layer)
    speech_layer = Dropout(0.2)(speech_layer)
    speech_layer = Reshape((-1, 640))(speech_layer)

    speech_layer = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2))(speech_layer)
    speech_layer = Dropout(0.2)(speech_layer)
    speech_layer = Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2))(speech_layer)
    speech_layer = Dropout(0.2)(speech_layer)
    speech_layer = AttentionDecoder(256, 256)(speech_layer)
    speech_layer = Flatten()(speech_layer)
    speech_layer = Dense(256, activation="relu")(speech_layer)

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

    combined_layer = concatenate([speech_layer, mocap_layer])
    combined_layer = Dense(256, activation='relu')(combined_layer)

    output_layer = Dense(4, activation='softmax')(combined_layer)

    model = Model(inputs=[speech_input_layer, mocap_input_layer], outputs=output_layer)

    metrics = top_k_accuracy()
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False, clipnorm=3.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=metrics)

    return model