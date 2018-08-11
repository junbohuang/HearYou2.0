
# coding: utf-8

# In[1]:


import os
import sys
import csv
import wave
import copy
import math

import numpy as np
import pandas as pd

from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import StratifiedKFold, KFold, train_test_split
from sklearn.svm import OneClassSVM, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.layers import LSTM, Input
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, Adam, RMSprop

sys.path.append("../")

from utilities.utils import *

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.display import clear_output


# In[2]:


batch_size = 64
nb_feat = 34
nb_class = 4
nb_epoch = 80

optimizer = 'Adadelta'


# In[3]:


params = Constants()
print(params)


# In[4]:


import pickle
with open(params.path_to_data + '/../'+'data_collected.pickle', 'rb') as handle:
    data2 = pickle.load(handle)


# In[5]:


data2[0]


# In[ ]:


import pickle
with open(params.path_to_data + '/../'+'data_collected.pickle', 'rb') as handle:
    data2 = pickle.load(handle)


# In[6]:


x_train2 = []
from sklearn.preprocessing import normalize
counter = 0
for ses_mod in data2:
    x_head = ses_mod['signal']
    st_features = calc_feat.calculate_features(x_head, params.framerate, None)
    st_features, _ = pad_sequence_into_array(st_features, maxlen=78)
    x_train2.append( st_features.T )
    counter+=1
    if(counter%100==0):
        print(counter)
    
x_train2 = np.array(x_train2)


# In[15]:


#x_train2 = x_train2.reshape(-1,78,34)
x_train2.shape


# In[47]:


def build_simple_lstm(nb_feat, nb_class, optimizer='Adadelta'):
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(78, nb_feat)))
    #model.add(Activation('tanh'))
    model.add(LSTM(256, return_sequences=False))
    #model.add(Activation('tanh'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


# In[48]:


model = build_simple_lstm(nb_feat, nb_class)
model.summary()


# In[49]:


Y=[]
for ses_mod in data2:
    Y.append(ses_mod['emotion'])
    
Y = to_categorical(Y)

Y.shape


# In[50]:


hist = model.fit(x_train2, Y, 
                 batch_size=batch_size, nb_epoch=80, verbose=1, shuffle = True, 
                 validation_split=0.2)


# In[63]:


from features import stFeatureExtraction

    
def calculate_features2(frames, freq, options):
    window_sec = 0.02
    window_n = int(freq * window_sec)
    use_derivatives = False

    st_f = stFeatureExtraction(frames, freq, window_n, window_n / 2)
    #print (st_f.shape)
    if st_f.shape[1] > 2:
        i0 = 1
        i1 = st_f.shape[1] - 1
        if i1 - i0 < 1:
            i1 = i0 + 1
        if use_derivatives:
            deriv_st_f = np.zeros((st_f.shape[0]*3, i1 - i0), dtype=float)
        else:
            deriv_st_f = np.zeros((st_f.shape[0], i1 - i0), dtype=float)
        for i in range(i0, i1):
            i_left = i - 1
            i_right = i + 1
            deriv_st_f[:st_f.shape[0], i - i0] = st_f[:, i]
            if use_derivatives:
                if st_f.shape[1] >= 2:
                    deriv_st_f[st_f.shape[0]:st_f.shape[0]*2, i - i0] = (st_f[:, i_right] - st_f[:, i_left]) / 2.
                    deriv_st_f[st_f.shape[0]*2:st_f.shape[0]*3, i - i0] =                         st_f[:, i] - 0.5*(st_f[:, i_left] + st_f[:, i_right])
        return deriv_st_f
    elif st_f.shape[1] == 2:
        deriv_st_f = np.zeros((st_f.shape[0], 1), dtype=float)
        deriv_st_f[:st_f.shape[0], 0] = st_f[:, 0]
        if use_derivatives:
            deriv_st_f[st_f.shape[0]:st_f.shape[0]*2, 0] = st_f[:, 1] - st_f[:, 0]
            deriv_st_f[st_f.shape[0]*2:st_f.shape[0]*3, 0] = np.zeros(st_f.shape[0])
        return deriv_st_f
    else:
        deriv_st_f = np.zeros((st_f.shape[0], 1), dtype=float)
        deriv_st_f[:st_f.shape[0], 0] = st_f[:, 0]
        if use_derivatives:
            deriv_st_f[st_f.shape[0]:st_f.shape[0]*2, 0] = np.zeros(st_f.shape[0])
            deriv_st_f[st_f.shape[0]*2:st_f.shape[0]*3, 0] = np.zeros(st_f.shape[0])
        return deriv_st_f


# In[64]:


x_train_sig2 = []
from sklearn.preprocessing import normalize
counter = 0
for ses_mod in data2:
    x_sig = ses_mod['signal']
    sig_avg = np.array_split(np.array(x_sig), 10)
    sig_feat = []
    for spl in sig_avg:
        #print(spl.shape)
        #print (x_sig.shape)
        st_features = calculate_features2(spl, params.framerate, None)
        st_features, _ = pad_sequence_into_array(st_features, maxlen=50)
        sig_feat.append( st_features ) 
    #st_features = calc_feat.calculate_features(x_sig, params.framerate, None)
    #st_features, _ = pad_sequence_into_array(st_features, maxlen=78)
    #x_train2.append( st_features.T )
    counter+=1
    if(counter%100==0):
        print(counter)
    x_train_sig2.append(np.array(sig_feat))
    #break
    
x_train_sig2 = np.array(x_train_sig2)
x_train_sig2.shape


# In[123]:


x_train3 = x_train2.reshape(-1,50,340)
x_train3.shape


# In[128]:


def build_simple_lstm2(nb_feat, nb_class, optimizer='SGD'):
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(50,340)))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


# In[129]:


model = build_simple_lstm2(nb_feat, nb_class)
model.summary()


# In[130]:


Y=[]
for ses_mod in data2:
    Y.append(ses_mod['emotion'])
    
Y = to_categorical(Y)

Y.shape


# In[131]:


hist = model.fit(x_train3, Y, 
                 batch_size=batch_size, nb_epoch=80, verbose=1, shuffle = True, 
                 validation_split=0.2)


# In[132]:


from keras.layers import Embedding, Conv2D, Dense, Dropout, Reshape, Merge, BatchNormalization, TimeDistributed, Lambda, Activation, LSTM, Flatten, Convolution2D, GRU, MaxPooling1D


def build_simple_lstm3(nb_feat, nb_class, optimizer='Adam'):
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(50,340)))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(512)))
    model.add(Activation('relu'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


# In[133]:


model = build_simple_lstm2(nb_feat, nb_class)
model.summary()


# In[134]:


hist = model.fit(x_train3, Y, 
                 batch_size=batch_size, nb_epoch=80, verbose=1, shuffle = True, 
                 validation_split=0.2)


# In[138]:


x_train4 = []
from sklearn.preprocessing import normalize
counter = 0
for ses_mod in data2:
    x_head = ses_mod['signal']
    st_features = calculate_features2(x_head, params.framerate, None)
    st_features, _ = pad_sequence_into_array(st_features, maxlen=400)
    x_train4.append( st_features )
    counter+=1
    if(counter%100==0):
        print(counter)
    #break
    
x_train4 = np.array(x_train4)
x_train4.shape


# In[152]:


x_train5 = x_train4.reshape(-1,34,400)


# In[178]:


def build_simple_lstm3(nb_feat, nb_class, optimizer='Adadelta'):
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(34,400)))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(512)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


# In[179]:


model = build_simple_lstm3(nb_feat, nb_class)
model.summary()


# In[180]:


hist = model.fit(x_train5, Y, 
                 batch_size=batch_size, nb_epoch=30, verbose=1, shuffle = True, 
                 validation_split=0.2)


# In[7]:


text = []


for ses_mod in data2:
    text.append(ses_mod['transcription'])


# In[10]:


from os import listdir
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, Reshape, Merge, BatchNormalization, TimeDistributed, Lambda, Activation, LSTM, Flatten, Convolution1D, GRU, MaxPooling1D
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
#from keras import initializers
from keras import backend as K
from keras.optimizers import SGD
from keras.optimizers import Adadelta
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras import optimizers
import numpy as np


# In[11]:


MAX_SEQUENCE_LENGTH = 500

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)

token_tr_X = tokenizer.texts_to_sequences(text)
x_train_text = []

x_train_text = sequence.pad_sequences(token_tr_X, maxlen=MAX_SEQUENCE_LENGTH)


# In[12]:


import codecs
EMBEDDING_DIM = 300

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

file_loc = params.path_to_data + '../glove.42B.300d.txt'

print (file_loc)

gembeddings_index = {}
with codecs.open(file_loc, encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        gembedding = np.asarray(values[1:], dtype='float32')
        gembeddings_index[word] = gembedding
#
f.close()
print('G Word embeddings:', len(gembeddings_index))

nb_words = len(word_index) +1
g_word_embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    gembedding_vector = gembeddings_index.get(word)
    if gembedding_vector is not None:
        g_word_embedding_matrix[i] = gembedding_vector
        
print('G Null word embeddings: %d' % np.sum(np.sum(g_word_embedding_matrix, axis=1) == 0))


# In[13]:


Y=[]
for ses_mod in data2:
    Y.append(ses_mod['emotion'])
    
Y = to_categorical(Y)

Y.shape


# In[14]:


model_text = Sequential()
#model.add(Embedding(2737, 128, input_length=MAX_SEQUENCE_LENGTH))
model_text.add(Embedding(nb_words,
                    EMBEDDING_DIM,
                    weights = [g_word_embedding_matrix],
                    input_length = MAX_SEQUENCE_LENGTH,
                    trainable = True))
model_text.add(Convolution1D(256, 3, border_mode='same'))
model_text.add(Dropout(0.2))
model_text.add(Activation('relu'))
model_text.add(Convolution1D(128, 3, border_mode='same'))
model_text.add(Dropout(0.2))
model_text.add(Activation('relu'))
model_text.add(Convolution1D(64, 3, border_mode='same'))
model_text.add(Dropout(0.2))
model_text.add(Activation('relu'))
model_text.add(Convolution1D(32, 3, border_mode='same'))
model_text.add(Dropout(0.2))
model_text.add(Activation('relu'))
model_text.add(Flatten())
model_text.add(Dropout(0.2))
model_text.add(Dense(256))


model_speech = Sequential()
model_speech.add(LSTM(512, return_sequences=True, input_shape=(78, nb_feat)))

model_speech.add(LSTM(256, return_sequences=False))

model_speech.add(Dense(512))

model_combined = Sequential()
model_combined.add(Merge([model_text, model_speech], mode='concat'))

model_combined.add(Dense(256))
model_combined.add(Activation('relu'))

model_combined.add(Dense(4))
model_combined.add(Activation('softmax'))

#sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model_combined.compile(loss='categorical_crossentropy',optimizer='Adadelta' ,metrics=['acc'])

## compille it here according to instructions

#model.compile()
model_combined.summary()

print("Model1 Built")


# In[17]:


hist = model_combined.fit([x_train_text,x_train2], Y, 
                 batch_size=batch_size, nb_epoch=30, verbose=1, 
                 validation_split=0.2)


# In[26]:



model_text2 = Sequential()
#model.add(Embedding(2737, 128, input_length=MAX_SEQUENCE_LENGTH))
model_text2.add(Embedding(nb_words,
                    EMBEDDING_DIM,
                    weights = [g_word_embedding_matrix],
                    input_length = MAX_SEQUENCE_LENGTH,
                    trainable = True))

model_text2.add(LSTM(512, return_sequences=True))
model_text2.add(Dropout(0.2))
model_text2.add(LSTM(256, return_sequences=False))
model_text2.add(Dropout(0.2))
model_text2.add(Dense(512))


model_speech = Sequential()
model_speech.add(LSTM(512, return_sequences=True, input_shape=(78, nb_feat)))
model_speech.add(Dropout(0.2))
model_speech.add(LSTM(256, return_sequences=False))
model_speech.add(Dropout(0.2))
model_speech.add(Dense(512))

model_combined = Sequential()
model_combined.add(Merge([model_text2, model_speech], mode='concat'))
model_combined.add(Dropout(0.2))
model_combined.add(Dense(256))
model_combined.add(Activation('relu'))

model_combined.add(Dense(4))
model_combined.add(Activation('softmax'))

#sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model_combined.compile(loss='categorical_crossentropy',optimizer='Adadelta' ,metrics=['acc'])

## compille it here according to instructions

#model.compile()
model_combined.summary()

print("Model1 Built")


# In[28]:


hist = model_combined.fit([x_train_text,x_train2], Y, 
                 batch_size=batch_size, nb_epoch=30, verbose=1, 
                 validation_split=0.2)


# In[84]:


model_speech = Sequential()
model_speech.add(Flatten(input_shape=(78, 34)))
model_speech.add(Dense(1024))
model_speech.add(Activation('relu'))
model_speech.add(Dropout(0.5))
model_speech.add(Dense(512))
model_speech.add(Activation('relu'))
model_speech.add(Dropout(0.5))
model_speech.add(Dense(256))
model_speech.add(Activation('relu'))
model_speech.add(Dense(4))
model_speech.add(Activation('softmax'))

#sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model_speech.compile(loss='categorical_crossentropy',optimizer='adam' ,metrics=['acc'])
model_speech.summary()


# In[46]:


hist = model_speech.fit(x_train2, Y, 
                 batch_size=batch_size, nb_epoch=80, verbose=1, shuffle = True, 
                 validation_split=0.2)


# In[51]:


model_text = Sequential()
#model.add(Embedding(2737, 128, input_length=MAX_SEQUENCE_LENGTH))
model_text.add(Embedding(nb_words,
                    EMBEDDING_DIM,
                    weights = [g_word_embedding_matrix],
                    input_length = MAX_SEQUENCE_LENGTH,
                    trainable = True))
model_text.add(Convolution1D(256, 3, border_mode='same'))
model_text.add(Dropout(0.2))
model_text.add(Activation('relu'))
model_text.add(Convolution1D(128, 3, border_mode='same'))
model_text.add(Dropout(0.2))
model_text.add(Activation('relu'))
model_text.add(Convolution1D(64, 3, border_mode='same'))
model_text.add(Dropout(0.2))
model_text.add(Activation('relu'))
model_text.add(Convolution1D(32, 3, border_mode='same'))
model_text.add(Dropout(0.2))
model_text.add(Activation('relu'))
model_text.add(Flatten())
model_text.add(Dropout(0.2))
model_text.add(Dense(256))


model_speech = Sequential()
model_speech.add(Flatten(input_shape=(78, 34)))
model_speech.add(Dense(1024))
model_speech.add(Activation('relu'))
model_speech.add(Dropout(0.5))
model_speech.add(Dense(512))
model_speech.add(Activation('relu'))
model_speech.add(Dropout(0.5))
model_speech.add(Dense(256))


model_combined = Sequential()
model_combined.add(Merge([model_text, model_speech], mode='concat'))

model_combined.add(Dense(256))
model_combined.add(Activation('relu'))

model_combined.add(Dense(4))
model_combined.add(Activation('softmax'))

#sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model_combined.compile(loss='categorical_crossentropy',optimizer='Adam' ,metrics=['acc'])

## compille it here according to instructions

#model.compile()
model_speech.summary()
model_text.summary()
model_combined.summary()

print("Model1 Built")


# In[52]:


hist = model_combined.fit([x_train_text,x_train2], Y, 
                 batch_size=batch_size, nb_epoch=30, verbose=1, 
                 validation_split=0.2)


# In[60]:


x_train_mocap = []
from sklearn.preprocessing import normalize
counter = 0
for ses_mod in data2:
    x_head = ses_mod['mocap_head']
    if(x_head.shape != (200,18)):
        x_head = np.zeros((200,18))   
    x_head[np.isnan(x_head)]=0
    x_hand = ses_mod['mocap_hand']
    if(x_hand.shape != (200,6)):
        x_hand = np.zeros((200,6))   
    x_hand[np.isnan(x_hand)]=0
    x_rot = ses_mod['mocap_rot']
    if(x_rot.shape != (200,165)):
        x_rot = np.zeros((200,165))   
    x_rot[np.isnan(x_rot)]=0
    #x_normed = (x - x.min(0)) / x.ptp(0)
    #x_normed = x_normed - 0.5
    #x_normed[np.isnan(x)]=0
    x_mocap = np.concatenate((x_head, x_hand), axis=1)
    x_mocap = np.concatenate((x_mocap, x_rot), axis=1)
    x_train_mocap.append( x_mocap )
    
x_train_mocap = np.array(x_train_mocap)
x_train_mocap = x_train_mocap.reshape(-1,200,189,1)
x_train_mocap.shape


# In[61]:


from keras.layers import Conv2D

model_text = Sequential()
#model.add(Embedding(2737, 128, input_length=MAX_SEQUENCE_LENGTH))
model_text.add(Embedding(nb_words,
                    EMBEDDING_DIM,
                    weights = [g_word_embedding_matrix],
                    input_length = MAX_SEQUENCE_LENGTH,
                    trainable = True))
model_text.add(Convolution1D(256, 3, border_mode='same'))
model_text.add(Dropout(0.2))
model_text.add(Activation('relu'))
model_text.add(Convolution1D(128, 3, border_mode='same'))
model_text.add(Dropout(0.2))
model_text.add(Activation('relu'))
model_text.add(Convolution1D(64, 3, border_mode='same'))
model_text.add(Dropout(0.2))
model_text.add(Activation('relu'))
model_text.add(Convolution1D(32, 3, border_mode='same'))
model_text.add(Dropout(0.2))
model_text.add(Activation('relu'))
model_text.add(Flatten())
model_text.add(Dropout(0.2))
model_text.add(Dense(256))


model_speech = Sequential()
model_speech.add(Flatten(input_shape=(78, 34)))
model_speech.add(Dense(1024))
model_speech.add(Activation('relu'))
model_speech.add(Dropout(0.5))
model_speech.add(Dense(512))
model_speech.add(Activation('relu'))
model_speech.add(Dropout(0.5))
model_speech.add(Dense(256))

model_mocap = Sequential()
model_mocap.add(Conv2D(32, 3, strides=(2, 2), border_mode='same', input_shape=(200, 189, 1)))
model_mocap.add(Dropout(0.2))
model_mocap.add(Activation('relu'))
model_mocap.add(Conv2D(64, 3, strides=(2, 2), border_mode='same'))
model_mocap.add(Dropout(0.2))
model_mocap.add(Activation('relu'))
model_mocap.add(Conv2D(64, 3, strides=(2, 2), border_mode='same'))
model_mocap.add(Dropout(0.2))
model_mocap.add(Activation('relu'))
model_mocap.add(Conv2D(128, 3, strides=(2, 2), border_mode='same'))
model_mocap.add(Dropout(0.2))
model_mocap.add(Activation('relu'))
model_mocap.add(Conv2D(128, 3, strides=(2, 2), border_mode='same'))
model_mocap.add(Dropout(0.2))
model_mocap.add(Activation('relu'))
model_mocap.add(Flatten())
model_mocap.add(Dense(256))

model_combined = Sequential()
model_combined.add(Merge([model_text, model_speech, model_mocap], mode='concat'))

model_combined.add(Dense(256))
model_combined.add(Activation('relu'))

model_combined.add(Dense(4))
model_combined.add(Activation('softmax'))

#sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model_combined.compile(loss='categorical_crossentropy',optimizer='Adam' ,metrics=['acc'])

## compille it here according to instructions

#model.compile()
model_speech.summary()
model_text.summary()
model_mocap.summary()
model_combined.summary()

print("Model1 Built")


# In[62]:


hist = model_combined.fit([x_train_text,x_train2,x_train_mocap], Y, 
                 batch_size=batch_size, nb_epoch=30, verbose=1, 
                 validation_split=0.2)


# In[65]:


x_train_sig2.shape


# In[70]:


model_speech = Sequential()
model_speech.add(Flatten(input_shape=(10, 34, 50)))
model_speech.add(Dense(4096))
model_speech.add(Activation('relu'))
model_speech.add(Dropout(0.25))
model_speech.add(Dense(1024))
model_speech.add(Activation('relu'))
model_speech.add(Dropout(0.25))
model_speech.add(Dense(256))
model_speech.add(Activation('relu'))
model_speech.add(Dense(4))
model_speech.add(Activation('softmax'))

#sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model_speech.compile(loss='categorical_crossentropy',optimizer='adadelta' ,metrics=['acc'])
model_speech.summary()


# In[71]:


hist = model_speech.fit(x_train_sig2, Y, 
                 batch_size=batch_size, nb_epoch=80, verbose=1, shuffle = True, 
                 validation_split=0.2)


# In[76]:


counter = 0
for ses_mod in data2:
    if (ses_mod['id'][:5]=="Ses05"):
        break
    counter+=1
counter


# In[82]:


xtrain_sp = x_train2[:3838]
xtest_sp = x_train2[3838:]
ytrain_sp = Y[:3838]
ytest_sp = Y[3838:]


# In[88]:


hist = model_speech.fit(xtrain_sp, ytrain_sp, 
                 batch_size=batch_size, nb_epoch=80, verbose=1, shuffle = True, 
                 validation_data=(xtest_sp, ytest_sp))


# In[89]:


x_train_long = []
from sklearn.preprocessing import normalize
counter = 0
for ses_mod in data2:
    x_head = ses_mod['signal']
    st_features = calc_feat.calculate_features(x_head, params.framerate, None)
    st_features, _ = pad_sequence_into_array(st_features, maxlen=125)
    x_train_long.append( st_features.T )
    counter+=1
    if(counter%100==0):
        print(counter)
    
x_train_long = np.array(x_train_long)


# In[90]:


x_train_long.shape


# In[91]:


model_speech = Sequential()
model_speech.add(Flatten(input_shape=(125, 34)))
model_speech.add(Dense(1024))
model_speech.add(Activation('relu'))
model_speech.add(Dropout(0.5))
model_speech.add(Dense(512))
model_speech.add(Activation('relu'))
model_speech.add(Dropout(0.5))
model_speech.add(Dense(256))
model_speech.add(Activation('relu'))
model_speech.add(Dense(4))
model_speech.add(Activation('softmax'))

#sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model_speech.compile(loss='categorical_crossentropy',optimizer='adam' ,metrics=['acc'])
model_speech.summary()


# In[92]:


hist = model_speech.fit(x_train_long, Y, 
                 batch_size=batch_size, nb_epoch=125, verbose=1, shuffle = True, 
                 validation_split = 0.2)


# In[95]:


from keras.layers import Conv2D

model_text = Sequential()
#model.add(Embedding(2737, 128, input_length=MAX_SEQUENCE_LENGTH))
model_text.add(Embedding(nb_words,
                    EMBEDDING_DIM,
                    weights = [g_word_embedding_matrix],
                    input_length = MAX_SEQUENCE_LENGTH,
                    trainable = True))
model_text.add(Convolution1D(256, 3, border_mode='same'))
model_text.add(Dropout(0.2))
model_text.add(Activation('relu'))
model_text.add(Convolution1D(128, 3, border_mode='same'))
model_text.add(Dropout(0.2))
model_text.add(Activation('relu'))
model_text.add(Convolution1D(64, 3, border_mode='same'))
model_text.add(Dropout(0.2))
model_text.add(Activation('relu'))
model_text.add(Convolution1D(32, 3, border_mode='same'))
model_text.add(Dropout(0.2))
model_text.add(Activation('relu'))
model_text.add(Flatten())
model_text.add(Dropout(0.2))
model_text.add(Dense(256))


model_speech = Sequential()
model_speech.add(Flatten(input_shape=(125, 34)))
model_speech.add(Dense(1024))
model_speech.add(Activation('relu'))
model_speech.add(Dropout(0.2))
model_speech.add(Dense(512))
model_speech.add(Activation('relu'))
model_speech.add(Dropout(0.2))
model_speech.add(Dense(256))

model_mocap = Sequential()
model_mocap.add(Conv2D(32, 3, strides=(2, 2), border_mode='same', input_shape=(200, 189, 1)))
model_mocap.add(Dropout(0.2))
model_mocap.add(Activation('relu'))
model_mocap.add(Conv2D(64, 3, strides=(2, 2), border_mode='same'))
model_mocap.add(Dropout(0.2))
model_mocap.add(Activation('relu'))
model_mocap.add(Conv2D(64, 3, strides=(2, 2), border_mode='same'))
model_mocap.add(Dropout(0.2))
model_mocap.add(Activation('relu'))
model_mocap.add(Conv2D(128, 3, strides=(2, 2), border_mode='same'))
model_mocap.add(Dropout(0.2))
model_mocap.add(Activation('relu'))
model_mocap.add(Conv2D(128, 3, strides=(2, 2), border_mode='same'))
model_mocap.add(Dropout(0.2))
model_mocap.add(Activation('relu'))
model_mocap.add(Flatten())
model_mocap.add(Dense(256))

model_combined = Sequential()
model_combined.add(Merge([model_text, model_speech, model_mocap], mode='concat'))

model_mocap.add(Dropout(0.2))

model_combined.add(Dense(256))
model_combined.add(Activation('relu'))

model_combined.add(Dense(4))
model_combined.add(Activation('softmax'))

#sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model_combined.compile(loss='categorical_crossentropy',optimizer='Adam' ,metrics=['acc'])

## compille it here according to instructions

#model.compile()
model_speech.summary()
model_text.summary()
model_mocap.summary()
model_combined.summary()

print("Model1 Built")


# In[96]:


hist = model_combined.fit([x_train_text,x_train_long,x_train_mocap], Y, 
                 batch_size=batch_size, nb_epoch=60, verbose=1, 
                 validation_split=0.2)

