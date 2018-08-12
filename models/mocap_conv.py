from keras.layers import Input, Conv2D, Dropout, Flatten, BatchNormalization, Dense
from keras.models import Model
from metrics.top_k_accuracy import *

## BLOODY EXPERIENCE!
## DO NOT ADD DENSE LAYER AFTER FLATTENING!!!!!

def load():


    input_layer = Input(shape=(200, 189, 1))
    layer = input_layer

    layer = Conv2D(32, 3, strides=(2,2), padding='same', activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.2)(layer)
    layer = Conv2D(64, 3, strides=(2,2), padding='same', activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.2)(layer)
    layer = Conv2D(64, 3, strides=(2,2), padding='same', activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.2)(layer)
    layer = Conv2D(128, 3, strides=(2,2), padding='same', activation='relu')(layer)
    layer = BatchNormalization()(layer)
    layer = Dropout(0.2)(layer)
    layer = Dropout(0.2)(layer)
    layer = Flatten()(layer)

    output_layer = Dense(4, activation='softmax')(layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    metrics = top_k_accuracy()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=metrics)

    return model
