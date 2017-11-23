import numpy as np
def compute_melgram(audio_path):
    

    # mel-spectrogram parameters
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  # to make it 1366 frame..

    src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA*SR)
    src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA*SR)
    print(n_sample,n_sample_fit)
    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA*SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample-n_sample_fit)/2:(n_sample+n_sample_fit)/2]

 
   
    
    logam = librosa.logamplitude
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                        n_fft=N_FFT, n_mels=N_MELS)**2,
                ref_power=1.0)
    ret = ret[np.newaxis, np.newaxis, :]
    return ret


from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense, Dropout, Reshape, Permute
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import GRU
from keras.utils.data_utils import get_file



def MusicTaggerCRNN(weights='msd', input_tensor=None,
                    include_top=True):
    
    import tensorflow as tf
    tf.python.control_flow_ops = tf
    tf.merge_all_summaries = tf

    
    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (1, 96, 1366)
    else:
        input_shape = (96, 1366, 1)

    if input_tensor is None:
        melgram_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            melgram_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            melgram_input = input_tensor

    # Determine input axis
    if K.image_dim_ordering() == 'th':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2

    
    x = ZeroPadding2D(padding=(0, 37))(melgram_input)
    x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(x)


    x01 = Convolution2D(32, 3, 3, border_mode='same')(x01)
    x01 = BatchNormalization(axis=channel_axis, mode=0)(x01)
    x01 = ELU()(x01)
    x02 = Convolution2D(32, 3, 3, border_mode='same')(x01)
    x02 = BatchNormalization(axis=channel_axis, mode=0)(x02)
    x02 = ELU()(x02)
    x03 = Convolution2D(32, 3, 3, border_mode='same')(x02)
    x03 = BatchNormalization(axis=channel_axis, mode=0)(x03)
    x03 = ELU()(x03)
    x03=x03+x01 #adding to maintain resnet property
    x03 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x03)
    x03 = Dropout(0.1)(x03)
    x04=x03
    x03=Convolution2D(64, 1, 1, border_mode='same')(x03)


    x11 = Convolution2D(64, 3, 3, border_mode='same')(x04)
    x11 = BatchNormalization(axis=channel_axis, mode=0)(x11)
    x11 = ELU()(x11)
    x12 = Convolution2D(64, 3, 3, border_mode='same')(x11)
    x12 = BatchNormalization(axis=channel_axis, mode=0)(x12)
    x12 = ELU()(x12)
    x12=x12+x03
    x13 = Convolution2D(64, 3, 3, border_mode='same')(x12)
    x13 = BatchNormalization(axis=channel_axis, mode=0)(x13)
    x13 = ELU()(x13)
    x13 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x13)
    x13 = Dropout(0.1)(x13)
    x12=Convolution2D(128, 1, 1, border_mode='same')(x12)


    x21 = Convolution2D(128, 3, 3, border_mode='same')(x12)
    x21 = BatchNormalization(axis=channel_axis, mode=0)(x21)
    x21 = ELU()(x12)
    x21=x12+x21
    x22 = Convolution2D(128, 3, 3, border_mode='same')(x21)
    x22 = BatchNormalization(axis=channel_axis, mode=0)(x22)
    x22 = ELU()(x22)
    x23 = Convolution2D(128, 3, 3, border_mode='same')(x22)
    x23 = BatchNormalization(axis=channel_axis, mode=0)(x23)
    x23 = ELU()(x23)
    x13 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x13)
    x23=x23+x21
    x23 = Dropout(0.1)(x23)
    
    
    x23 = Reshape((6, 170))(x23)

    # GRU block 1, 2, output
    x23 = GRU(32, return_sequences=True, name='gru1')(x23)
    x23 = GRU(32, return_sequences=False, name='gru2')(x23)
    x = Dropout(0.3)(x23)
    if include_top:
        x = Dense(50, activation='sigmoid', name='output')(x)

    
    model = Model(melgram_input, x)     
    model.load_weights('d:/music_tagger_crnn_weights_tensorflow.h5')
    return model


    

    
import time
import numpy as np
from keras import backend as K

import pdb

def sort_result(tags, preds):
    result = zip(tags, preds)
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)
    return [(name, '%5.3f' % score) for name, score in sorted_result]



    


    
    
audio_paths = ['D:/music-auto_tagging-keras-master/data/1.mp3','D:/music-auto_tagging-keras-master/data/2.mp3']
melgram_paths = ['D:/music-auto_tagging-keras-master/data/1.npy','D:/music-auto_tagging-keras-master/data/2.npy']


tags = ['rock', 'pop', 'alternative', 'indie', 'electronic',
            'female vocalists', 'dance', '00s', 'alternative rock', 'jazz',
            'beautiful', 'metal', 'chillout', 'male vocalists',
            'classic rock', 'soul', 'indie rock', 'Mellow', 'electronica',
            '80s', 'folk', '90s', 'chill', 'instrumental', 'punk',
            'oldies', 'blues', 'hard rock', 'ambient', 'acoustic',
            'experimental', 'female vocalist', 'guitar', 'Hip-Hop',
            '70s', 'party', 'country', 'easy listening',
            'sexy', 'catchy', 'funk', 'electro', 'heavy metal',
            'Progressive rock', '60s', 'rnb', 'indie pop',
            'sad', 'House', 'happy']

    # prepare data like this   __import__('librosa')


    
melgrams = np.zeros((0,1, 96,1366))
    
    
for melgram_path in melgram_paths:
            melgram = np.load(melgram_path)
            print(melgram.shape,"tf")
            print(melgram)
            melgrams = np.concatenate((melgrams, melgram), axis=0)

    
    
model = MusicTaggerCRNN(weights=None)
model.summary()
   
print('Predicting... with melgrams: ', melgrams.shape)
start = time.time()
pred_tags = model.predict(melgrams)
print("Prediction is done. The AUC is..." ,time.time()-start-2.5)
print('Printing top-10 tags for each track...')
for song_idx, audio_path in enumerate(audio_paths):
        sorted_result = sort_result(tags, pred_tags[song_idx, :].tolist())
        print(audio_path)
        print(sorted_result[:10])
        print(sorted_result[10:20])
        print(' ')

   

