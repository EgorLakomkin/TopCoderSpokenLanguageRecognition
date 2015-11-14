import librosa
import numpy as np
from librosa.feature import delta, melspectrogram
import time,os
from scipy.io.wavfile import read
from sklearn.mixture import GMM
from utils import get_mfcc
from vad import remove_silence
import time
import traceback, sys
from sklearn.preprocessing import StandardScaler

fft_points = 1024

fft_overlap = int(fft_points*0.5)
mfcc_coefficients = 13

gmm_fft_points = 1024
gmm_fft_overlap = int(gmm_fft_points*0.5)
gmm_mfcc_coefficients = 13

import copy
def preprocess_mfcc(mfcc):
    mfcc_cp = copy.deepcopy(mfcc)
    for i in xrange(mfcc.shape[1]):
        mfcc_cp[:,i] = mfcc[:,i] - np.mean(mfcc[:,i])
        mfcc_cp[:,i] = mfcc_cp[:,i]/np.max(np.abs(mfcc_cp[:,i]))
    return mfcc_cp

def extract_feature_spectrogram( data ):
    filename, lbl = data
    try:
        signal,sr = librosa.load(filename)
        if len(signal) < 5*sr:
            print "Too short"
            return filename, None, None
        else:
            print "OK", len(signal) / float(sr)
        if sr != 22050:
            print "Non standart sr", sr
            return filename, None, None
        signal = signal[:5*sr]
        spectrogram = melspectrogram(y=signal, sr=sr, n_fft=fft_points,hop_length = fft_overlap,
                                    fmax=5000, n_mels = mfcc_coefficients)
    #spectrogram = spectrogram / spectrogram.max()
    #print gmm.means_.shape
    #result_features = np.vstack( [ gmm. ] )
        return filename, lbl, spectrogram
    except Exception,e:
        print e
        return filename, None, None

def extract_mfcc_features(data, max_length_sec = 10 ):
    try:
        filename, lbl = data
        #signal, sr = librosa.load(filename)
        sr, signal = read(filename)
        if len(signal) == 0:
            return filename, None, None
        if len(signal.shape) > 1:
            signal = signal[:,0]
        signal = signal - signal.mean() 
        signal = signal[:max_length_sec*sr]
        signal = np.array(remove_silence( list(signal), 0.01 ))
        if np.sum(signal) == 0.0:
            print "Empty", filename
            return filename, None, None
        mfcc = librosa.feature.mfcc( signal, n_fft = fft_points, hop_length = fft_overlap, n_mfcc = mfcc_coefficients, fmax = 5000 )
        delta_mfcc_1 = delta( mfcc, order = 1 )
        delta_mfcc_2 = delta( mfcc, order = 2 )
        #print "Took", time.time() - start, "length", original_len, "size", os.path.getsize( filename ), "pre process", preprocess_time, "load", loading_time
        total_features = np.vstack( [ mfcc, delta_mfcc_1, delta_mfcc_2 ] )
        total_features = np.transpose( total_features )
        total_features = preprocess_mfcc( total_features )
        #total_features = StandardScaler().fit_transform( total_features )
        return filename, lbl, total_features
    except Exception,e:
        print signal, signal.shape
        print e
        traceback.print_exc(file=sys.stdout)
        print filename
        return filename, None, None

def extract_gmm_feature( data, max_length_sec = 10 ):
    try:
        filename, lbl = data
        sr,signal = read(filename)
        if len(signal.shape) > 1:
            signal = signal[:,0]

        signal = signal - signal.mean()
        signal = signal[:max_length_sec*sr]
        signal = np.array(remove_silence( signal, 0.005 ))
        if np.sum(signal) == 0.0:
            print "Empty", filename
            return filename, None, None

        mfcc = librosa.feature.mfcc( signal, n_fft = gmm_fft_points, hop_length = gmm_fft_overlap, n_mfcc = gmm_mfcc_coefficients, fmax = 5000 )
        #mfcc = preprocess_mfcc(mfcc)
        delta_mfcc_1 = delta( mfcc, order = 1 )
        delta_mfcc_2 = delta( mfcc, order = 2 )
        total_features = np.vstack( [ mfcc, delta_mfcc_1, delta_mfcc_2 ] )
        total_features = np.transpose( total_features )
        total_features = preprocess_mfcc( total_features )        
        #total_features = StandardScaler().fit_transform( total_features )
        gmm = GMM(n_components=1)
        gmm.fit( total_features )
        res_features = np.hstack( [gmm.means_[0], gmm.covars_[0]] )
        #print gmm.means_.shape
        #result_features = np.vstack( [ gmm. ] )
        return filename, lbl, res_features
    except Exception,e:
        print e
        return filename, None, None
