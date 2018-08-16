import wave
import numpy as np
import importlib
import pickle

import cv2
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from tensorflow.python import debug as tf_debug

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

from sklearn.preprocessing import label_binarize
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

from features import *
from plotters.confusion_matrix import *
from callbacks.ConfusionMatrixLogger import ConfusionMatrixPlotter
from callbacks.AccuracyLossLogger import AccLossPlotter


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



def train(config, model, xtrain, ytrain, xtest, ytest):

    epochs = config['epochs']
    batch_size = config['batch_size']
    model_name = config['model'].split('.')[-1]
    emotion_class = config['emotion']
    validation_split = np.array(config['train_val_test_split'])[1] / \
                       (np.array(config['train_val_test_split'])[1] +
                        np.array(config['train_val_test_split'])[0])

    path_to_plots = './plots/' + model_name
    if not os.path.exists(path_to_plots):
        os.makedirs(path_to_plots)

    path_to_log = './logs/' + model_name
    if not os.path.exists(path_to_log):
        os.makedirs(path_to_log)

    csv_name = path_to_log + '/' + model_name + '.log'
    csv_logger = CSVLogger(csv_name)

    accloss_logger = AccLossPlotter(model_name)

    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(ytrain.argmax(1)),
                                                      ytrain.argmax(1))
    class_weights_dict = {}
    for n in range(len(class_weights)):
        class_weights_dict[n] = class_weights[n]

    # tensorboard = TensorBoard(log_dir=path_to_log, histogram_freq=1, write_graph=False, write_grads=False)
    # cm_logger = ConfusionMatrixPlotter(xtrain, ytrain, emotion_class, model_name)

    ## FOR SAVING MODEL
    save_path = os.path.join(path_to_log, model_name) + '.h5'
    # check_pointer = ModelCheckpoint(save_path, save_best_only=True)

    ## FOR DEBUG
    # K.set_session(
    #     tf_debug.TensorBoardDebugWrapperSession(
    #         tf.Session(), "127.0.0.1:8080"))
    # K.get_session().run(tf.global_variables_initializer())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        model.fit(xtrain, ytrain,
                  batch_size=batch_size, epochs=epochs, verbose=1,
                  validation_split=validation_split, shuffle=True,
                  callbacks=[csv_logger, accloss_logger],
                  class_weight=class_weights_dict)
        print("trained. saving model...")
        model.save_weights(save_path)
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
        plot_cm(model_name, emotion_class, ytest, prediction)
        print("confusion matrix saved!")

# def evaluate(config, model, xtest, ytest):
#     model_name = config['model'].split('.')[-1]
#     emotion_class = config['emotion']
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         model.load_weights(save_path)
#         print("evaluating...")
#         scores = model.evaluate(x=xtest, y=ytest, verbose=0)
#         for n in range(len(scores)):
#             if n == 0:
#                 print("%s: %.3f" % (model.metrics_names[n], scores[n]))
#             else:
#                 print("%s: %.2f%%" % (model.metrics_names[n], scores[n] * 100))
#
#         print("predicting...")
#         prediction = model.predict(xtest, verbose=0)
#     plot_cm(model_name, emotion_class, ytest, prediction)
#     print("confusion matrix saved!")

def dir_setup():
    if not os.path.exists('./plots'):
        os.mkdir('./plots')
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    if not os.path.exists('./datasets'):
        os.mkdir('./datasets')


# def make_movie(images, path, width, height):
#
#     video_name = os.path.join(path, 'evolutionary_confusion_matrix.mp4')
#
#     video = cv2.VideoWriter(filename=video_name,
#                             fourcc=-1,
#                             fps=3,
#                             frameSize=(width,height),
#                             isColor=True)
#
#     for image in images:
#         video.write(cv2.imread(os.path.join(path, image)))
#
#     video.release()
#     cv2.destroyAllWindows()

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
    train_val_test_split = np.array(config['train_val_test_split'])
    split_seed = config['split_seed']
    code_path = os.path.dirname(os.path.realpath(os.getcwd()))
    test_size = train_val_test_split[-1]

    with open(code_path + '/HearYou2.0/datasets/data_collected.pickle', 'rb') as handle:
        data2 = pickle.load(handle)

    if model_name == 'text_speech_mocap' or model_name == 'text_speech_mocap_attention':
        nb_words, g_word_embedding_matrix, x_train_text = get_transcription(data2)
        x_train_speech = get_speech_features(data2)
        x_train_mocap = get_mocap(data2)
        Y = get_label(data2, emotions_used)

        xtrain_sp, xtest_sp, ytrain, ytest= train_test_split(x_train_speech, Y,
                                                                   test_size=test_size,
                                                                   random_state=split_seed,
                                                                   shuffle=True)

        xtrain_tx, xtest_tx, _, _ = train_test_split(x_train_text, Y,
                                                            test_size=test_size,
                                                            random_state=split_seed,
                                                            shuffle=True)

        xtrain_mo, xtest_mo, _, _ = train_test_split(x_train_mocap, Y,
                                                     test_size=test_size,
                                                     random_state=split_seed,
                                                     shuffle=True)

        xtrain = [xtrain_tx, xtrain_sp, xtrain_mo]
        xtest = [xtest_tx, xtest_sp, xtest_mo]

        return xtrain, ytrain, xtest, ytest, nb_words, g_word_embedding_matrix

    if model_name == 'text_speech' or model_name == 'text_speech_attention':
        nb_words, g_word_embedding_matrix, x_train_text = get_transcription(data2)
        x_train_speech = get_speech_features(data2)
        Y = get_label(data2, emotions_used)

        xtrain_sp, xtest_sp, ytrain, ytest = train_test_split(x_train_speech, Y,
                                                               test_size=test_size,
                                                               random_state=split_seed,
                                                               shuffle=True)

        xtrain_tx, xtest_tx, _, _ = train_test_split(x_train_text, Y,
                                                     test_size=test_size,
                                                     random_state=split_seed,
                                                     shuffle=True)

        xtrain = [xtrain_tx, xtrain_sp]
        xtest = [xtest_tx, xtest_sp]

        return xtrain, ytrain, xtest, ytest, nb_words, g_word_embedding_matrix

    if model_name == 'text_lstm' or model_name == 'text_attention':
        nb_words, g_word_embedding_matrix, x_train_text = get_transcription(data2)
        Y = get_label(data2, emotions_used)

        xtrain_tx, xtest_tx, ytrain, ytest = train_test_split(x_train_text, Y,
                                                     test_size=test_size,
                                                     random_state=split_seed,
                                                     shuffle=True)

        xtrain = xtrain_tx
        xtest = xtest_tx

        return xtrain, ytrain, xtest, ytest, nb_words, g_word_embedding_matrix

    if model_name == 'speech_dense' or model_name == 'speech_lstm' or model_name == 'speech_lstm_attention':
        x_train_speech = get_speech_features(data2)
        Y = get_label(data2, emotions_used)

        xtrain_sp, xtest_sp, ytrain, ytest = train_test_split(x_train_speech, Y,
                                                               test_size=test_size,
                                                               random_state=split_seed,
                                                               shuffle=True)

        xtrain = xtrain_sp
        xtest = xtest_sp

        return xtrain, ytrain, xtest, ytest

    if model_name == 'speech_mocap' or model_name == 'speech_mocap_attention':
        x_train_speech = get_speech_features(data2)
        x_train_mocap = get_mocap(data2)
        Y = get_label(data2, emotions_used)

        xtrain_sp, xtest_sp, ytrain, ytest = train_test_split(x_train_speech, Y,
                                                               test_size=test_size,
                                                               random_state=split_seed,
                                                               shuffle=True)
        xtrain_mo, xtest_mo, _, _ = train_test_split(x_train_mocap, Y,
                                                     test_size=test_size,
                                                     random_state=split_seed,
                                                     shuffle=True)

        xtrain = [xtrain_sp, xtrain_mo]
        xtest = [xtest_sp, xtest_mo]

        return xtrain, ytrain, xtest, ytest

    if model_name == 'mocap_conv':
        x_train_mocap = get_mocap(data2)
        Y = get_label(data2, emotions_used)

        xtrain_mo, xtest_mo, ytrain, ytest = train_test_split(x_train_mocap, Y,
                                                     test_size=test_size,
                                                     random_state=split_seed,
                                                     shuffle=True)

        xtrain = xtrain_mo
        xtest = xtest_mo

        return xtrain, ytrain, xtest, ytest

    if model_name == 'mocap_lstm' or 'mocap_lstm_attention':
        x_train_mocap = get_mocap(data2)
        Y = get_label(data2, emotions_used)

        x_train_mocap = x_train_mocap.reshape(-1, 200, 189)
        xtrain_mo, xtest_mo, ytrain, ytest = train_test_split(x_train_mocap, Y,
                                                               test_size=test_size,
                                                               random_state=split_seed,
                                                               shuffle=True)

        xtrain = xtrain_mo
        xtest = xtest_mo

        return xtrain, ytrain, xtest, ytest

    else:
        raise NotImplementedError

def load_model(config):
    module_model = config['model']
    print("model:", module_model)
    module = importlib.import_module(module_model)
    if 'text' in module_model:
        xtrain, ytrain, xtest, ytest, nb_words, g_word_embedding_matrix = feed_data(config)
        model = module.load(nb_words, g_word_embedding_matrix)
    else:
        print(config)
        xtrain, ytrain, xtest, ytest = feed_data(config)
        model = module.load()
    return model, xtrain, ytrain, xtest, ytest


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