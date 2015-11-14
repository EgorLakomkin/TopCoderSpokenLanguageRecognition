import os
from multiprocessing import Pool
from features_storage import FeatureStorage
from features import extract_feature_spectrogram, extract_mfcc_features

VOXFORGE_DIR = '/store/egor/voxforge'
from utils import FEATURE_STORAGE, load_classes_info
from path import Path
import random
import cPickle as pickle
import cPickle
from cPickle import Unpickler

random.seed(100500)

def extract_vox_files_by_dir(dir):
    files = []
    d = Path(dir)
    seen_dirs = set()
    for f in d.walkfiles('*_processed.wav'):
        file_dir = str(f.splitpath()[0])
        if file_dir not in seen_dirs:
            files.append( str(f) )
            seen_dirs.add( file_dir )
    return files


def get_voxforge_total_files( vox_forge_dir, max = 3000):
    CLASSES_DICT, _ = load_classes_info()
    dirs = {'Italian' : os.path.join(vox_forge_dir, 'it' ),
            'French'  : os.path.join(vox_forge_dir, 'fr' ),
            'German'  : os.path.join(vox_forge_dir, 'de' ),
            'English' : os.path.join(vox_forge_dir, 'en' ),
            'Spanish' : os.path.join(vox_forge_dir, 'es' )
    }

    available_data = []
    for lang, path in dirs.iteritems():
        files = extract_vox_files_by_dir( path )
        random.shuffle(files)
        files = files[:max]
        print "Available {} files for language {}".format( lang, len(files) )
        lang_class = CLASSES_DICT[ lang ]
        available_data.extend( [ (f, lang_class) for f in files ] )
    return available_data

def load_vox_forge_files(vox_dir, max_files, vox_file = 'voxforge.pcl', feature_func = extract_mfcc_features):
    feature_storage = FeatureStorage( name = vox_file, base_dir = FEATURE_STORAGE )
    if feature_storage.exists():
        print "Loading voxforge from pickle"
        return feature_storage.load()
    else:
        pool = Pool()
        
        files = get_voxforge_total_files(vox_dir, max_files)
        print "Started processing voxforge data {}".format( len(files) )
        voxforge_features = pool.map( feature_func, files )

        pool.close()
        pool.terminate()
        print "Dumping voxforge features"
        feature_storage.save( voxforge_features )
        return voxforge_features

