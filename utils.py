import os
from subprocess import check_call
from path import Path
import librosa
from cPickle import Pickler, Unpickler
import cPickle
from multiprocessing import Pool
import numpy as np
from features_storage import FeatureStorage

DATA_DIR = './data'
FEATURE_STORAGE = 'features_storage'

TRAINING_FILE = os.path.join(DATA_DIR, 'trainingset.csv')
TESTING_FILE = os.path.join(DATA_DIR, 'testingset.csv' )

FILES_DIR = os.path.join( DATA_DIR, 'data' )


def load_classes_info():
    classes_dict, revert_classes_dict = {}, {}
    class_idx = 0
    for line in open(TRAINING_FILE):
        _, language_class_name = line.strip().split(',')
        if language_class_name not in classes_dict:
            classes_dict[ language_class_name ] = class_idx
            revert_classes_dict[ class_idx ] = language_class_name
            class_idx += 1
    return classes_dict, revert_classes_dict

def get_mfcc(signal, n_fft = 4096, hop_length = 1024, sr=44100, n_mfcc=20, logscaled=True):
  """Computes the mel-frequency cepstral coefficients of a signal
    ARGS
      signal: audio signal <number array>
      n_fft: FFT size <int>
      hop_length : hop length <int>    
      sr: sampling rate <int>
      n_mfcc: number of MFC coefficients <int>
      logscaled: log-scale the magnitudes of the spectrogram <bool>
    RETURN
      mfcc: mel-frequency cepstral coefficients  <number numpy array>
  """ 
  S = librosa.feature.melspectrogram(signal, sr=sr, n_fft=n_fft, hop_length=hop_length)
  if logscaled:
    log_S = librosa.logamplitude(S)
  mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=n_mfcc)
  return mfcc


def load_test_features(test_filename, feature_func, limit = None):
    pool = Pool()
    test_data = get_all_test_data()
    if limit:
        test_data = test_data[:limit]
    feature_storage = FeatureStorage( name = test_filename, base_dir = FEATURE_STORAGE )
    if not  feature_storage.exists():
        print "Loading test from scratch"
        for_features = [ (path, None)  for (path, filename ) in test_data  ]
        X_test_transformed = pool.map( feature_func, for_features )
        print "Dumping test features"


        feature_storage.save( X_test_transformed )
        print "Finished dumping"
    else:
       print "Loading test from cache"
       X_test_transformed =  feature_storage.load()
    pool.close()
    pool.terminate()
    return X_test_transformed


def shuffle_in_unison_inplace(a, b):
    assert a.shape[0] == b.shape[0]
    p = np.random.permutation(a.shape[0])
    return a[p], b[p]


def load_train_features( train_filename, feature_func, limit = None ):

    feature_storage = FeatureStorage( name = train_filename, base_dir = FEATURE_STORAGE )
    if not feature_storage.exists(  ):

        all_train_data = return_all_train_files()
        if limit is not None:
            all_train_data = all_train_data[:limit]
        print "Started processing train"
        pool = Pool()
        X_train_transformed = pool.map( feature_func, all_train_data )
        pool.close()
        pool.terminate()
        print "Dumping train features"
        feature_storage.save( X_train_transformed )
    else:
        print "Loading train from cache"
        X_train_transformed = feature_storage.load()

    return X_train_transformed


def convert_to_wav(dir):
    train_dir = Path( dir )
    for f in train_dir.walkfiles('*.mp3'):
        name = f.name.replace('.mp3', '') + '.wav'
        check_call(['avconv', '-ar', '44100', '-i', str(f), os.path.abspath( os.path.join( dir, name ) )])



def get_all_test_data():
    res = []
    for line in open(TESTING_FILE):
        filename = line.strip()
        full_path = os.path.join(FILES_DIR, filename.replace('.mp3', '.wav'))
        res.append( (full_path, filename) )
    return res

def return_all_train_files():
    all_train_data = []
    classes_dict, _ = load_classes_info()
    for line in open(TRAINING_FILE):
        filename, language_class_name = line.strip().split(',')
        filename = os.path.join(FILES_DIR, filename.replace('.mp3', '.wav'))
        language_class = classes_dict[ language_class_name ]
        all_train_data.append( (filename, language_class) )
    return all_train_data


if __name__ == "__main__":
    convert_to_wav('./data')
