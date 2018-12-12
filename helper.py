import wave
import numpy as np
import importlib
import pickle

import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, EarlyStopping

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from sklearn.preprocessing import label_binarize
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

from features import *
from plotters.confusion_matrix import *
from callbacks.ConfusionMatrixLogger import ConfusionMatrixPlotter
from callbacks.AccuracyLossLogger import AccLossPlotter



def get_path(*_args):
    if len(_args) == 2:
        path = os.path.join(_args[0], _args[1])
    elif len(_args) == 1:
        path = os.path.join(os.getcwd(), _args[0])
    else:
        raise Exception("Unexpected Argument")

    return path
    
def detect_gender(utterance):
    if utterance['id'][-4] == "M":
        gender = "male"
    else:
        gender = "female"
    return gender


def detect_session(utterance):
    session = utterance['id'][4]
    return "ses" + session

def detect_acting_type(utterance):
    if utterance['id'][7:12] == "impro":
        acting_type = "improvised"
    else:
        acting_type = "scripted"

    return acting_type


def count_utterance(data):
    utterance_count_dict = {
        "ses1": {
            "scripted": {"male": 0, "female": 0},
            "improvised": {"male": 0, "female": 0}},
        "ses2": {
            "scripted": {"male": 0, "female": 0},
            "improvised": {"male": 0, "female": 0}},
        "ses3": {
            "scripted": {"male": 0, "female": 0},
            "improvised": {"male": 0, "female": 0}},
        "ses4": {
            "scripted": {"male": 0, "female": 0},
            "improvised": {"male": 0, "female": 0}},
        "ses5": {
            "scripted": {"male": 0, "female": 0},
            "improvised": {"male": 0, "female": 0}}}

    for utt in data:
        ses = detect_session(utt)
        acting_type = detect_acting_type(utt)
        gender = detect_gender(utt)
        utterance_count_dict[ses][acting_type][gender] += 1

    return utterance_count_dict


def get_count(utterance_count_dict, data_type):
    num_train = 0
    num_test = 0

    for ses in ["ses1", "ses2", "ses3", "ses4"]:
        if data_type == "improvised" or data_type == "scripted":
            num_train += sum(utterance_count_dict[ses][data_type].values())
            num_test = sum(utterance_count_dict["ses5"][data_type].values())

        elif data_type == "all":
            num_train += sum(utterance_count_dict[ses]["scripted"].values()) + sum(
                utterance_count_dict[ses]["improvised"].values())
            num_test = sum(utterance_count_dict["ses5"]["scripted"].values()) + sum(
                utterance_count_dict["ses5"]["improvised"].values())

        else:
            raise NotImplementedError(data_type + " is beyond the data type...")

    return num_train, num_test


def dir_setup():
    if not os.path.exists('./plots'):
        os.mkdir('./plots')
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    if not os.path.exists('./datasets'):
        os.mkdir('./datasets')
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')


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


def get_transcription(data, data_type="improvised"):
    code_path = os.path.dirname(os.path.realpath(os.getcwd()))
    print("getting trarnscription...")
    text = []

    for utterance in data:
        if data_type == "all":
            text.append(utterance['transcription'])

        elif data_type == "scripted":
            if detect_acting_type(utterance) == "scripted":
                text.append(utterance['transcription'])

        elif data_type == "improvised":
            if detect_acting_type(utterance) == "improvised":
                text.append(utterance['transcription'])

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


def get_speech_features(data, data_type="improvised", feature_type="mfcc", mode="dynamic"):
    framerate = 16000
    eps = 1e-5
    print("creating features for speech...")
    print(data_type)
    print("mode", mode)
    utterance_count_dict = count_utterance(data)
    train_size, test_size = get_count(utterance_count_dict, data_type)

    if feature_type == "all":
        feature_size = 34
        feature_on = 0
    elif feature_type == "mfcc":
        feature_size = 12
        feature_on = 8
        feature_off = 20
    elif feature_type == "logmel":
        feature_size = 13
        feature_on = 8
        feature_off = 20
    else:
        raise NotImplementedError(feature_type + "is beyond us...")

    if mode == "dynamic":
        features = []
        deltas = []
        deltadeltas = []
        speech_features = np.zeros((train_size + test_size, 100, feature_size, 3))

        counter = 0

        for utterance in data:
            if data_type == "all":
                x_head = utterance['signal']
                st_features = calculate_features(x_head, framerate, None)
                delta = stDelta(st_features, 2)
                deltadelta = stDelta(delta, 2)
                st_features, _ = pad_sequence_into_array(st_features, maxlen=100)
                delta, _ = pad_sequence_into_array(delta, maxlen=100)
                deltadelta, _ = pad_sequence_into_array(deltadelta, maxlen=100)
                features.append(st_features.T)
                deltas.append(delta.T)
                deltadeltas.append(deltadelta.T)

                counter += 1

                if (counter % 500 == 0):
                    print(counter)

            elif data_type == "scripted":
                if detect_acting_type(utterance) == "scripted":
                    x_head = utterance['signal']
                    st_features = calculate_features(x_head, framerate, None)
                    delta = stDelta(st_features, 2)
                    deltadelta = stDelta(delta, 2)
                    st_features, _ = pad_sequence_into_array(st_features, maxlen=100)
                    delta, _ = pad_sequence_into_array(delta, maxlen=100)
                    deltadelta, _ = pad_sequence_into_array(deltadelta, maxlen=100)
                    features.append(st_features.T)
                    deltas.append(delta.T)
                    deltadeltas.append(deltadelta.T)

                    counter += 1

                    if (counter % 500 == 0):
                        print(counter)

            elif data_type == "improvised":
                if detect_acting_type(utterance) == "improvised":
                    x_head = utterance['signal']
                    st_features = calculate_features(x_head, framerate, None)
                    delta = stDelta(st_features, 2)
                    deltadelta = stDelta(delta, 2)
                    st_features, _ = pad_sequence_into_array(st_features, maxlen=100)
                    delta, _ = pad_sequence_into_array(delta, maxlen=100)
                    deltadelta, _ = pad_sequence_into_array(deltadelta, maxlen=100)
                    features.append(st_features.T)
                    deltas.append(delta.T)
                    deltadeltas.append(deltadelta.T)

                    counter += 1

                    if (counter % 500 == 0):
                        print(counter)
            else:
                raise ValueError(data_type + "is beyond us...")

        # feature normalization
        features = np.array(features)
        deltas = np.array(deltas)
        deltadeltas = np.array(deltadeltas)

        if feature_type == "mfcc":
            features = features[:, :, feature_on:feature_off]
            deltas = deltas[:, :, feature_on:feature_off]
            deltadeltas = deltadeltas[:, :, feature_on:feature_off]

        mean_features = np.mean(features[:train_size], axis=0)
        std_features = np.std(features[:train_size], axis=0)
        mean_deltas = np.mean(deltas[:train_size], axis=0)
        std_deltas = np.std(deltas[:train_size], axis=0)
        mean_deltadeltas = np.mean(deltadeltas[:train_size], axis=0)
        std_deltadeltas = np.std(deltadeltas[:train_size], axis=0)

        speech_features[:, :, :, 0] = (features[:, :, :] - mean_features) / (std_features + eps)
        speech_features[:, :, :, 1] = (features[:, :, :] - mean_deltas) / (std_deltas + eps)
        speech_features[:, :, :, 2] = (features[:, :, :] - mean_deltadeltas) / (std_deltadeltas + eps)
        print("speech_features shape", speech_features.shape)

    if mode == "static":
        features = []
        speech_features = np.zeros((train_size + test_size, 100, feature_size))

        counter = 0

        for utterance in data:
            if data_type == "all":
                x_head = utterance['signal']
                st_features = calculate_features(x_head, framerate, None)
                st_features, _ = pad_sequence_into_array(st_features, maxlen=100)
                features.append(st_features.T)

                counter += 1

                if (counter % 500 == 0):
                    print(counter)

            elif data_type == "scripted":
                if detect_acting_type(utterance) == "scripted":
                    x_head = utterance['signal']
                    st_features = calculate_features(x_head, framerate, None)
                    st_features, _ = pad_sequence_into_array(st_features, maxlen=100)
                    features.append(st_features.T)

                    counter += 1

                    if (counter % 500 == 0):
                        print(counter)

            elif data_type == "improvised":
                if detect_acting_type(utterance) == "improvised":
                    x_head = utterance['signal']
                    st_features = calculate_features(x_head, framerate, None)
                    st_features, _ = pad_sequence_into_array(st_features, maxlen=100)
                    features.append(st_features.T)

                    counter += 1

                    if (counter % 500 == 0):
                        print(counter)
            else:
                raise ValueError(data_type + "is beyond us...")

        # feature normalization
        features = np.array(features)

        if feature_type == "mfcc":
            features = features[:, :, feature_on:feature_off]

        mean_features = np.mean(features[:train_size], axis=0)
        std_features = np.std(features[:train_size], axis=0)

        speech_features[:, :, :] = (features[:, :, :] - mean_features) / (std_features + eps)
        print("speech_features shape", speech_features.shape)

    #else:
      #  raise NotImplementedError("we only do static or dynamic bro!")

    return speech_features


def get_mocap(data, data_type="improvised"):
    print("creating mocap data...")
    x_train_mocap = []
    for utterance in data:
        if data_type == "all":
            x_head = utterance['mocap_head']
            if (x_head.shape != (200, 18)):
                x_head = np.zeros((200, 18))
            x_head[np.isnan(x_head)] = 0
            x_hand = utterance['mocap_hand']
            if (x_hand.shape != (200, 6)):
                x_hand = np.zeros((200, 6))
            x_hand[np.isnan(x_hand)] = 0
            x_rot = utterance['mocap_rot']
            if (x_rot.shape != (200, 165)):
                x_rot = np.zeros((200, 165))
            x_rot[np.isnan(x_rot)] = 0
            x_mocap = np.concatenate((x_head, x_hand), axis=1)
            x_mocap = np.concatenate((x_mocap, x_rot), axis=1)
            x_train_mocap.append(x_mocap)
        elif data_type == "scripted":
            if detect_acting_type(utterance) == "scripted":
                x_head = utterance['mocap_head']
                if (x_head.shape != (200, 18)):
                    x_head = np.zeros((200, 18))
                x_head[np.isnan(x_head)] = 0
                x_hand = utterance['mocap_hand']
                if (x_hand.shape != (200, 6)):
                    x_hand = np.zeros((200, 6))
                x_hand[np.isnan(x_hand)] = 0
                x_rot = utterance['mocap_rot']
                if (x_rot.shape != (200, 165)):
                    x_rot = np.zeros((200, 165))
                x_rot[np.isnan(x_rot)] = 0
                x_mocap = np.concatenate((x_head, x_hand), axis=1)
                x_mocap = np.concatenate((x_mocap, x_rot), axis=1)
                x_train_mocap.append(x_mocap)
        elif data_type == "improvised":
            if detect_acting_type(utterance) == "improvised":
                x_head = utterance['mocap_head']
                if (x_head.shape != (200, 18)):
                    x_head = np.zeros((200, 18))
                x_head[np.isnan(x_head)] = 0
                x_hand = utterance['mocap_hand']
                if (x_hand.shape != (200, 6)):
                    x_hand = np.zeros((200, 6))
                x_hand[np.isnan(x_hand)] = 0
                x_rot = utterance['mocap_rot']
                if (x_rot.shape != (200, 165)):
                    x_rot = np.zeros((200, 165))
                x_rot[np.isnan(x_rot)] = 0
                x_mocap = np.concatenate((x_head, x_hand), axis=1)
                x_mocap = np.concatenate((x_mocap, x_rot), axis=1)
                x_train_mocap.append(x_mocap)

    x_train_mocap = np.array(x_train_mocap)
    x_train_mocap = x_train_mocap.reshape(-1, 200, 189, 1)
    print("x_train_mocap.shape",x_train_mocap.shape)

    return x_train_mocap


def get_label(data, emotions_used, data_type="improvised"):
    print("creating labels...")
    Y = []
    for utterance in data:
        if data_type == "all":
            Y.append(utterance['emotion'])
        elif data_type == "scripted":
            if detect_acting_type(utterance) == "scripted":
                Y.append(utterance['emotion'])
        elif data_type == "improvised":
            if detect_acting_type(utterance) == "improvised":
                Y.append(utterance['emotion'])

    Y = label_binarize(Y, emotions_used)

    print("Y.shape: ", Y.shape)

    return Y


def feed_data(config):
    model_name = config['model'].split('.')[-1]
    emotions_used = np.array(config['emotion'])
    code_path = os.path.dirname(os.path.realpath(os.getcwd()))
    data_type = config['data_type']
    feature_type = config['feature_type']
    mode = config['mode']

    with open(code_path + '/HearYou2.0/datasets/data_collected.pickle', 'rb') as handle:
        data = pickle.load(handle)

    utterance_count_dict = count_utterance(data)
    train_size, test_size = get_count(utterance_count_dict, data_type)

    print("Training on", model_name)
    print("Data type:", data_type)
    print("Feature type:", feature_type)
    print("Training data size:", train_size)
    print("Testing data size:", test_size)

    if model_name == 'text_speech_mocap' or model_name == 'text_speech_mocap_delta':
        nb_words, g_word_embedding_matrix, x_text = get_transcription(data, data_type=data_type)
        x_speech = get_speech_features(data, data_type=data_type, feature_type=feature_type, mode=mode)
        x_mocap = get_mocap(data, data_type=data_type)
        Y = get_label(data, emotions_used, data_type=data_type)

        # train test split
        xtrain_sp = x_speech[:train_size]
        xtest_sp = x_speech[train_size:]
        xtrain_tx = x_text[:train_size]
        xtest_tx = x_text[train_size:]
        xtrain_mo = x_mocap[:train_size]
        xtest_mo = x_mocap[train_size:]
        ytrain = Y[:train_size]
        ytest= Y[train_size:]

        xtrain = [xtrain_tx, xtrain_sp, xtrain_mo]
        xtest = [xtest_tx, xtest_sp, xtest_mo]

        return xtrain, ytrain, xtest, ytest, nb_words, g_word_embedding_matrix

    if model_name == 'text_speech' or model_name == 'text_speech_delta':
        nb_words, g_word_embedding_matrix, x_text = get_transcription(data, data_type=data_type)
        x_speech = get_speech_features(data, data_type=data_type, feature_type=feature_type, mode=mode)
        Y = get_label(data, emotions_used, data_type=data_type)

        # train test split
        xtrain_sp = x_speech[:train_size]
        xtest_sp = x_speech[train_size:]
        xtrain_tx = x_text[:train_size]
        xtest_tx = x_text[train_size:]
        ytrain = Y[:train_size]
        ytest = Y[train_size:]

        xtrain = [xtrain_tx, xtrain_sp]
        xtest = [xtest_tx, xtest_sp]

        return xtrain, ytrain, xtest, ytest, nb_words, g_word_embedding_matrix

    if model_name == 'text_lstm' or model_name == 'text_lstm_attention':
        nb_words, g_word_embedding_matrix, x_text = get_transcription(data, data_type=data_type)
        Y = get_label(data, emotions_used, data_type=data_type)

        # train test split
        xtrain_tx = x_text[:train_size]
        xtest_tx = x_text[train_size:]
        ytrain = Y[:train_size]
        ytest = Y[train_size:]

        xtrain = xtrain_tx
        xtest = xtest_tx

        return xtrain, ytrain, xtest, ytest, nb_words, g_word_embedding_matrix

    if model_name == 'speech_dense' or model_name == 'speech_lstm' or model_name == 'speech_lstm_attention' or model_name == 'speech_delta':
        x_speech = get_speech_features(data, data_type=data_type, feature_type=feature_type, mode=mode)
        Y = get_label(data, emotions_used, data_type=data_type)

        # train test split
        xtrain_sp = x_speech[:train_size]
        xtest_sp = x_speech[train_size:]
        ytrain = Y[:train_size]
        ytest = Y[train_size:]

        xtrain = xtrain_sp
        xtest = xtest_sp

        return xtrain, ytrain, xtest, ytest

    if model_name == 'speech_mocap_delta':
        x_speech = get_speech_features(data, data_type=data_type, feature_type=feature_type, mode=mode)
        x_mocap = get_mocap(data, data_type=data_type)
        Y = get_label(data, emotions_used, data_type=data_type)

        # train test split
        xtrain_sp = x_speech[:train_size]
        xtest_sp = x_speech[train_size:]
        xtrain_mo = x_mocap[:train_size]
        xtest_mo = x_mocap[train_size:]
        ytrain = Y[:train_size]
        ytest = Y[train_size:]

        xtrain = [xtrain_sp, xtrain_mo]
        xtest = [xtest_sp, xtest_mo]

        return xtrain, ytrain, xtest, ytest

    if model_name == 'mocap_conv':
        x_mocap = get_mocap(data, data_type=data_type)
        Y = get_label(data, emotions_used, data_type=data_type)

        # train test split
        xtrain_mo = x_mocap[:train_size]
        xtest_mo = x_mocap[train_size:]
        ytrain = Y[:train_size]
        ytest = Y[train_size:]

        xtrain = xtrain_mo
        xtest = xtest_mo

        return xtrain, ytrain, xtest, ytest

    if model_name == 'mocap_lstm' or 'mocap_lstm_attention':
        x_mocap = get_mocap(data, data_type=data_type)
        Y = get_label(data, emotions_used, data_type=data_type)
        x_mocap = x_mocap.reshape(-1, 200, 189)

        # train test split
        xtrain_mo = x_mocap[:train_size]
        xtest_mo = x_mocap[train_size:]
        ytrain = Y[:train_size]
        ytest = Y[train_size:]

        xtrain = xtrain_mo
        xtest = xtest_mo

        return xtrain, ytrain, xtest, ytest

    else:
        raise NotImplementedError

def load_model(config):

    model_name = config['model'].split('.')[-1]

    path_to_plots = './plots/' + model_name
    if not os.path.exists(path_to_plots):
        os.makedirs(path_to_plots)

    path_to_log = './logs/' + model_name
    if not os.path.exists(path_to_log):
        os.makedirs(path_to_log)

    if config["feature_type"] == "mfcc":
        feat_size = 12
    elif config["feature_type"] == "all":
        feat_size = 34
    module_model = config['model']
    print("model:", module_model)
    module = importlib.import_module(module_model)
    if 'text' in module_model:
        xtrain, ytrain, xtest, ytest, nb_words, g_word_embedding_matrix = feed_data(config)
        model = module.load(nb_words, g_word_embedding_matrix, feat_size)
    else:
        print(config)
        xtrain, ytrain, xtest, ytest = feed_data(config)
        model = module.load(feat_size)


    return model, xtrain, ytrain, xtest, ytest


def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''

    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return LearningRateScheduler(schedule)



def train(config, model, xtrain, ytrain, xtest, ytest):

    epochs = config['epochs']
    batch_size = config['batch_size']
    model_name = config['model'].split('.')[-1]
    emotion_class = config['emotion']
    data_type = config['data_type']
    feature_type = config['feature_type']

    validation_split = np.array(config['train_val_split'])[1] / \
                       (np.array(config['train_val_split'])[1] +
                        np.array(config['train_val_split'])[0])

    path_to_log = './logs/' + model_name

    csv_name = path_to_log + '/' + model_name + '_' + data_type + '_' + feature_type + '.log'
    csv_logger = CSVLogger(csv_name)

    accloss_logger = AccLossPlotter(model_name)

    class_weights_train = class_weight.compute_class_weight('balanced',
                                                      np.unique(ytrain.argmax(1)),
                                                      ytrain.argmax(1))
    class_weights_test = class_weight.compute_class_weight('balanced',
                                                      np.unique(ytest.argmax(1)),
                                                      ytest.argmax(1))
    class_weights_train_dict = {}
    for n in range(len(class_weights_train)):
        class_weights_train_dict[n] = class_weights_train[n]
    print("class_weights_train_dict", class_weights_train_dict)
    print("class counts for training data:")
    print(np.bincount(ytrain.argmax(1))[np.unique(ytrain.argmax(1))])

    class_weights_test_dict = {}
    for n in range(len(class_weights_test)):
        class_weights_test_dict[n] = class_weights_test[n]
    print("class_weights_test_dict", class_weights_test_dict)
    print("class counts for testing data:")
    print(np.bincount(ytest.argmax(1))[np.unique(ytest.argmax(1))])



    # tensorboard = TensorBoard(log_dir=path_to_log, histogram_freq=1, write_graph=False, write_grads=False)
    # cm_logger = ConfusionMatrixPlotter(xtrain, ytrain, emotion_class, model_name)
    lr_sched = step_decay_schedule(initial_lr=1e-4, decay_factor=0.75, step_size=4)

    ## FOR SAVING MODEL
    h5_name =  model_name + '_' + data_type + '_' + feature_type + '.h5'
    weights_save_path = os.path.join(path_to_log, h5_name)

    js_name = model_name + '_' + data_type + '_' + feature_type + '.json'
    archi_save_path = os.path.join(path_to_log, js_name)
    with open(archi_save_path, 'w') as f:
        f.write(model.to_json())

    # check_pointer = ModelCheckpoint(save_path, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=30)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        model.fit(xtrain, ytrain,
                    batch_size=batch_size, epochs=epochs, verbose=1,
                    validation_split=validation_split, shuffle=True,
                    callbacks=[lr_sched, early_stopping, accloss_logger],
                    class_weight=class_weights_train_dict)
        print("trained. saving model...")
        model.save_weights(weights_save_path)
        print("Saved model to disk")

        print("evaluating...")
        scores = model.evaluate(x=xtest, y=ytest, verbose=0)
        for n in range(len(scores)):
            if n == 0:
                print("%s: %.3f" % (model.metrics_names[n], scores[n]))
            else:
                print("%s: %.2f%%" % (model.metrics_names[n], scores[n] * 100))

        print("predicting...")
        prediction = model.predict(xtest, verbose=0)
        plot_cm(model_name, emotion_class, ytest, prediction, data_type, feature_type)
        print("confusion matrix saved!")

    return model
