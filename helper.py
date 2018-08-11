import os
import wave
import numpy as np
import importlib
import pickle
import itertools


from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence


from sklearn.preprocessing import label_binarize
from sklearn.utils import class_weight

from features import *
from callbacks.metrics_logger import MetricsLogger

def split_wav(wav, emotions):
    (nchannels, sampwidth, framerate, nframes, comptype, compname), samples = wav

    left = samples[0::nchannels]
    right = samples[1::nchannels]

    frames = []
    for ie, e in enumerate(emotions):
        start = e['start']
        end = e['end']

        e['right'] = right[int(start * framerate):int(end * framerate)]
        e['left'] = left[int(start * framerate):int(end * framerate)]

        frames.append({'left': e['left'], 'right': e['right']})
    return frames


def get_field(data, key):
    return np.array([e[key] for e in data])

def pad_sequence_into_array(Xs, maxlen=None, truncating='post', padding='post', value=0.):

    Nsamples = len(Xs)
    if maxlen is None:
        lengths = [s.shape[0] for s in Xs]    # 'sequences' must be list, 's' must be numpy array, len(s) return the first dimension of s
        maxlen = np.max(lengths)

    Xout = np.ones(shape=[Nsamples, maxlen] + list(Xs[0].shape[1:]), dtype=Xs[0].dtype) * np.asarray(value, dtype=Xs[0].dtype)
    Mask = np.zeros(shape=[Nsamples, maxlen], dtype=Xout.dtype)
    for i in range(Nsamples):
        x = Xs[i]
        if truncating == 'pre':
            trunc = x[-maxlen:]
        elif truncating == 'post':
            trunc = x[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % truncating)
        if padding == 'post':
            Xout[i, :len(trunc)] = trunc
            Mask[i, :len(trunc)] = 1
        elif padding == 'pre':
            Xout[i, -len(trunc):] = trunc
            Mask[i, -len(trunc):] = 1
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return Xout, Mask


def convert_gt_from_array_to_list(gt_batch, gt_batch_mask=None):

    B, L = gt_batch.shape
    gt_batch = gt_batch.astype('int')
    gts = []
    for i in range(B):
        if gt_batch_mask is None:
            l = L
        else:
            l = int(gt_batch_mask[i, :].sum())
        gts.append(gt_batch[i, :l].tolist())
    return gts

def get_audio(path_to_wav, filename):
    wav = wave.open(path_to_wav + filename, mode="r")
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
    content = wav.readframes(nframes)
    samples = np.fromstring(content, dtype=np.int16)
    return (nchannels, sampwidth, framerate, nframes, comptype, compname), samples


def get_transcriptions(path_to_transcriptions, filename):
    f = open(path_to_transcriptions + filename, 'r').read()
    f = np.array(f.split('\n'))
    transcription = {}
    for i in range(len(f) - 1):
        g = f[i]
        i1 = g.find(': ')
        i0 = g.find(' [')
        ind_id = g[:i0]
        ind_ts = g[i1+2:]
        transcription[ind_id] = ind_ts
    return transcription


def get_emotions(path_to_emotions, filename):
    f = open(path_to_emotions + filename, 'r').read()
    f = np.array(f.split('\n'))
    idx = f == ''
    idx_n = np.arange(len(f))[idx]
    emotion = []
    for i in range(len(idx_n) - 2):
        g = f[idx_n[i]+1:idx_n[i+1]]
        head = g[0]
        i0 = head.find(' - ')
        start_time = float(head[head.find('[') + 1:head.find(' - ')])
        end_time = float(head[head.find(' - ') + 3:head.find(']')])
        actor_id = head[head.find(filename[:-4]) + len(filename[:-4]) + 1:
                        head.find(filename[:-4]) + len(filename[:-4]) + 5]
        emo = head[head.find('\t[') - 3:head.find('\t[')]
        vad = head[head.find('\t[') + 1:]

        v = float(vad[1:7])
        a = float(vad[9:15])
        d = float(vad[17:23])
        
        j = 1
        emos = []
        while g[j][0] == "C":
            head = g[j]
            start_idx = head.find("\t") + 1
            evoluator_emo = []
            idx = head.find(";", start_idx)
            while idx != -1:
                evoluator_emo.append(head[start_idx:idx].strip().lower()[:3])
                start_idx = idx + 1
                idx = head.find(";", start_idx)
            emos.append(evoluator_emo)
            j += 1

        emotion.append({'start': start_time,
                        'end': end_time,
                        'id': filename[:-4] + '_' + actor_id,
                        'v': v,
                        'a': a,
                        'd': d,
                        'emotion': emo,
                        'emo_evo': emos})
    return emotion



def train(model_name, model, xtrain, ytrain, validation_data, batch_size, epochs):
    path_to_ckp = './logs/' + model_name
    if not os.path.exists(path_to_ckp):
        os.makedirs(path_to_ckp)
    # tensorboard = TensorBoard(log_dir=path_to_ckp, histogram_freq=1, write_graph=True, write_images=True,
    #                           write_grads=True)
    csv_name = path_to_ckp + '/' + model_name + '.log'
    csv_logger = CSVLogger(csv_name)
    # save_path = path_to_ckp + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    # check_pointer = ModelCheckpoint(save_path, save_best_only=True)
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(ytrain.argmax(1)),
                                                      ytrain.argmax(1))
    class_weights_dict = {}
    for n in range(len(class_weights)):
        class_weights_dict[n] = class_weights[n]

    return model.fit(xtrain, ytrain,
                     batch_size=batch_size, epochs=epochs, verbose=1,
                     validation_data=validation_data,
                     callbacks=[csv_logger],
                     class_weight=class_weights_dict)


def predict(model, validation_data):
    predictions = model.predict(validation_data[0], verbose=1)
    return predictions


def dir_setup():
    if not os.path.exists('./plots'):
        os.mkdir('./plots')
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    if not os.path.exists('./datasets'):
        os.mkdir('./datasets')


def get_transcription(data2):
    code_path = os.path.dirname(os.path.realpath(os.getcwd()))
    print("getting trarnscription...")
    text = []

    for ses_mod in data2:
        text.append(ses_mod['transcription'])

    MAX_SEQUENCE_LENGTH = 500

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)

    token_tr_X = tokenizer.texts_to_sequences(text)

    x_train_text = sequence.pad_sequences(token_tr_X, maxlen=MAX_SEQUENCE_LENGTH)

    # embeddings
    print("embedding...")
    import codecs

    EMBEDDING_DIM = 300

    word_index = tokenizer.word_index
    print('Found %s unique tokens' % len(word_index))

    file_loc = code_path + '/HearYou2.0/datasets/glove.42B.300d.txt'

    print(file_loc)

    gembeddings_index = {}
    with codecs.open(file_loc, encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            gembedding = np.asarray(values[1:], dtype='float32')
            gembeddings_index[word] = gembedding

    f.close()
    print('G Word embeddings:', len(gembeddings_index))

    nb_words = len(word_index) + 1
    g_word_embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        gembedding_vector = gembeddings_index.get(word)
        if gembedding_vector is not None:
            g_word_embedding_matrix[i] = gembedding_vector

    print('G Null word embeddings: %d' % np.sum(np.sum(g_word_embedding_matrix, axis=1) == 0))

    return nb_words, g_word_embedding_matrix, x_train_text


def get_speech_features(data2):
    framerate = 16000
    print("creating features for speech...")
    x_train_speech = []

    counter = 0
    for ses_mod in data2:
        x_head = ses_mod['signal']
        st_features = calculate_features(x_head, framerate, None)
        st_features, _ = pad_sequence_into_array(st_features, maxlen=100)
        x_train_speech.append(st_features.T)
        counter += 1
        if (counter % 100 == 0):
            print(counter)

    x_train_speech = np.array(x_train_speech)
    print("x_train_speech shape: ", x_train_speech.shape)

    return x_train_speech


def get_mocap(data2):
    print("creating mocap data...")
    x_train_mocap = []
    for ses_mod in data2:
        x_head = ses_mod['mocap_head']
        if (x_head.shape != (200, 18)):
            x_head = np.zeros((200, 18))
        x_head[np.isnan(x_head)] = 0
        x_hand = ses_mod['mocap_hand']
        if (x_hand.shape != (200, 6)):
            x_hand = np.zeros((200, 6))
        x_hand[np.isnan(x_hand)] = 0
        x_rot = ses_mod['mocap_rot']
        if (x_rot.shape != (200, 165)):
            x_rot = np.zeros((200, 165))
        x_rot[np.isnan(x_rot)] = 0
        x_mocap = np.concatenate((x_head, x_hand), axis=1)
        x_mocap = np.concatenate((x_mocap, x_rot), axis=1)
        x_train_mocap.append(x_mocap)

    x_train_mocap = np.array(x_train_mocap)
    print("x_train_mocap.shape", x_train_mocap.shape)
    x_train_mocap = x_train_mocap.reshape(-1, 200, 189, 1)
    print("x_train_mocap.shape",x_train_mocap.shape)

    return x_train_mocap


def get_label(data2, emotions_used):
    print("creating labels...")
    Y = []
    for ses_mod in data2:
        Y.append(ses_mod['emotion'])

    Y = label_binarize(Y, emotions_used)

    print("Y.shape: ",Y.shape)

    return Y

def feed_data(config):
    model_name = config['model'].split('.')[-1]
    emotions_used = np.array(config['emotion'])
    code_path = os.path.dirname(os.path.realpath(os.getcwd()))

    with open(code_path + '/HearYou2.0/datasets/data_collected.pickle', 'rb') as handle:
        data2 = pickle.load(handle)

    if model_name == 'text_speech_mocap':
        nb_words, g_word_embedding_matrix, x_train_text = get_transcription(data2)
        x_train_speech = get_speech_features(data2)
        x_train_mocap = get_mocap(data2)
        Y = get_label(data2, emotions_used)

        xtrain_sp = x_train_speech[:3838]
        xtest_sp = x_train_speech[3838:]
        xtrain_tx = x_train_text[:3838]
        xtest_tx = x_train_text[3838:]
        ytrain_sp = Y[:3838]
        ytest_sp = Y[3838:]
        xtrain_mo = x_train_mocap[:3838]
        xtest_mo = x_train_mocap[3838:]

        ytrain = ytrain_sp
        xtrain = [xtrain_tx, xtrain_sp, xtrain_mo]

        validation_data = ([xtest_tx, xtest_sp, xtest_mo], ytest_sp)
        return xtrain, ytrain, validation_data, nb_words, g_word_embedding_matrix

    if model_name == 'text_speech':
        nb_words, g_word_embedding_matrix, x_train_text = get_transcription(data2)
        x_train_speech = get_speech_features(data2)
        Y = get_label(data2, emotions_used)

        xtrain_sp = x_train_speech[:3838]
        xtest_sp = x_train_speech[3838:]
        xtrain_tx = x_train_text[:3838]
        xtest_tx = x_train_text[3838:]
        ytrain_sp = Y[:3838]
        ytest_sp = Y[3838:]

        ytrain = ytrain_sp
        xtrain = [xtrain_tx, xtrain_sp]

        validation_data = ([xtest_tx, xtest_sp], ytest_sp)
        return xtrain, ytrain, validation_data, nb_words, g_word_embedding_matrix

    if model_name == 'text_lstm':
        nb_words, g_word_embedding_matrix, x_train_text = get_transcription(data2)
        Y = get_label(data2, emotions_used)

        xtrain_tx = x_train_text[:3838]
        xtest_tx = x_train_text[3838:]
        ytrain_sp = Y[:3838]
        ytest_sp = Y[3838:]

        ytrain = ytrain_sp
        xtrain = xtrain_tx

        validation_data = (xtest_tx, ytest_sp)
        return xtrain, ytrain, validation_data, nb_words, g_word_embedding_matrix

    if model_name == 'speech_dense' or model_name == 'speech_lstm':
        x_train_speech = get_speech_features(data2)
        Y = get_label(data2, emotions_used)

        xtrain_sp = x_train_speech[:3838]
        xtest_sp = x_train_speech[3838:]
        ytrain_sp = Y[:3838]
        ytest_sp = Y[3838:]

        ytrain = ytrain_sp
        xtrain = xtrain_sp

        validation_data = (xtest_sp, ytest_sp)
        return xtrain, ytrain, validation_data

    if model_name == 'speech_mocap':
        x_train_speech = get_speech_features(data2)
        x_train_mocap = get_mocap(data2)
        Y = get_label(data2, emotions_used)

        xtrain_sp = x_train_speech[:3838]
        xtest_sp = x_train_speech[3838:]
        ytrain_sp = Y[:3838]
        ytest_sp = Y[3838:]
        xtrain_mo = x_train_mocap[:3838]
        xtest_mo = x_train_mocap[3838:]

        ytrain = ytrain_sp
        xtrain = [xtrain_sp, xtrain_mo]

        validation_data = ([xtest_sp, xtest_mo], ytest_sp)
        return xtrain, ytrain, validation_data

    if model_name == 'mocap_conv':
        x_train_mocap = get_mocap(data2)
        Y = get_label(data2, emotions_used)

        ytrain_sp = Y[:3838]
        ytest_sp = Y[3838:]
        xtrain_mo = x_train_mocap[:3838]
        xtest_mo = x_train_mocap[3838:]

        ytrain = ytrain_sp
        xtrain = xtrain_mo

        validation_data = (xtest_mo, ytest_sp)
        return xtrain, ytrain, validation_data

    if model_name == 'mocap_lstm':
        x_train_mocap = get_mocap(data2)
        Y = get_label(data2, emotions_used)

        x_train_mocap2 = x_train_mocap.reshape(-1, 200, 189)
        xtrain_mo2 = x_train_mocap2[:3838]
        xtest_mo2 = x_train_mocap2[3838:]
        ytrain_sp = Y[:3838]
        ytest_sp = Y[3838:]

        ytrain = ytrain_sp
        xtrain = xtrain_mo2
        validation_data = (xtest_mo2, ytest_sp)
        return xtrain, ytrain, validation_data


def load_model(module_model, config):
    module = importlib.import_module(module_model)
    if 'text' in module_model:
        xtrain, ytrain, validation_data, nb_words, g_word_embedding_matrix = feed_data(config)
        model = module.load(nb_words, g_word_embedding_matrix)
    else:
        xtrain, ytrain, validation_data = feed_data(config)
        model = module.load()
    return model, xtrain, ytrain, validation_data


def calculate_features(frames, freq, options):
    window_sec = 0.2
    window_n = int(freq * window_sec)

    st_f = stFeatureExtraction(frames, freq, window_n, window_n / 2)

    if st_f.shape[1] > 2:
        i0 = 1
        i1 = st_f.shape[1] - 1
        if i1 - i0 < 1:
            i1 = i0 + 1

        deriv_st_f = np.zeros((st_f.shape[0], i1 - i0), dtype=float)
        for i in range(i0, i1):
            i_left = i - 1
            i_right = i + 1
            deriv_st_f[:st_f.shape[0], i - i0] = st_f[:, i]
        return deriv_st_f
    elif st_f.shape[1] == 2:
        deriv_st_f = np.zeros((st_f.shape[0], 1), dtype=float)
        deriv_st_f[:st_f.shape[0], 0] = st_f[:, 0]
        return deriv_st_f
    else:
        deriv_st_f = np.zeros((st_f.shape[0], 1), dtype=float)
        deriv_st_f[:st_f.shape[0], 0] = st_f[:, 0]
        return deriv_st_f