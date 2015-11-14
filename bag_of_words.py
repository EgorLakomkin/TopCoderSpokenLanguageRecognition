#-*-coding:utf-8-*-
from collections import Counter
import os
import random
import cPickle as pickle
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.sparse import csr_matrix
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import itertools
import time
from get_voxforge_data  import load_vox_forge_files
from sofia_clustering import SofiaClusterer
from utils import load_train_features, load_test_features, load_classes_info
import cPickle
__author__ = 'egor'
from cPickle import Unpickler
from features import extract_mfcc_features
from utils import return_all_train_files, get_all_test_data
from multiprocessing import Pool
import numpy as np
import sys
import joblib
#import line_profiler

#profile = lambda f: f

def learn_dictionary(path, features, filename = 'dictionary.pcl', number_words = 1000, mini_batch_size = 10000, iterations = 10000):
    if not os.path.exists(filename):
        clusterer = MiniBatchKMeans(  n_clusters = number_words, verbose = 2,
reassignment_ratio = 10 ** -4,batch_size = mini_batch_size, compute_labels = True, n_init = 20  )
#    clusterer = SofiaClusterer(path, num_clusters = number_words, mini_batch_size = mini_batch_size, iterations = iterations )
        clusterer.fit( features )
        joblib.dump( clusterer, filename )
    else:
        print "Loading dictionary"
        clusterer = joblib.load( filename )
    clusterer.verbose = 0
    print "Dictionary loaded"
    return clusterer


def func_predictor(data):
    x, clusterer = data
    preds =  clusterer.predict(x)
    return preds

class BagOfMFCCTransformer(TransformerMixin):
     def __init__(self, clusterer, max_clusters = 1000, *args, **kwargs):
         self.clusterer = clusterer
         self.max_clusters = max_clusters

     def fit_transform(self, X, y=None, **fit_params):
        print "Fit transforming"
        self.fit(X, y, **fit_params)
        return self.transform(X)

     def transform(self, X, y=None):
         """
         (num_tracks , ( features ))
         """
         print "Transforming with dictionary"
         start = time.time()
         pool = Pool()
         classes = pool.map( func_predictor, [ (x,self.clusterer) for x in X ] )#[self.clusterer.predict( x ) for x in X ]
         pool.close()
         pool.terminate()
         print "Predicted classes"
         res = [np.bincount(c) for c in classes ]
         #
         print "bin count"

         res= csr_matrix([[0 if i >= len(track_classes) else track_classes[i]
                             for i in xrange(0, self.max_clusters) ] for track_classes in res ])
         print "total", time.time() - start
         return res

     def fit(self,X, y=None):
        return self




if __name__ == "__main__":
    
    LIMIT = 9000
    DICTIONARY_SIZE = 600
    sofia_path = None#sys.argv[1]
    if len(sys.argv) > 1:
      VOX_FEATURES = sys.argv[1] == "use_voxforge"
    else:
      VOX_FEATURES = False

    print "Using Vox features", VOX_FEATURES
    print "LIMIT", LIMIT

    print "Checking class names"
    _, REVERSE_CLASSES = load_classes_info()

    print "Loading train"
    X_train_transformed = load_train_features('bow_train_features.pcl', extract_mfcc_features, limit = LIMIT )
    X_train = np.vstack([f for (_,_,f) in X_train_transformed if f is not None])
    #X_train = np.reshape(X_train, ( len(X_train_transformed), num_features ) )
    print X_train.shape
    #for filename, lbl, features in X_train_transformed:
    #    if features is None:
    #        print "Train",filename, "is none"
    #        continue
    #    if X_total is not None:
    #        X_total =  np.vstack([ X_total, features ])
    #    else:
    #        X_total = features
    X_total = X_train
    print "Ready with train"
    print "Starting train"
    test_data = get_all_test_data()
    filename_by_path = {path : filename  for (path, filename) in test_data }
    X_test_transformed = load_test_features('bow_test_features.pcl', extract_mfcc_features, limit = LIMIT)
    
    X_test = np.vstack([f for (_,_,f) in X_test_transformed if f is not None])
    #for filename, lbl, features in X_test_transformed: 
    #    if features is not None:
    #        X_total = np.vstack([ X_total, features ])
    #    else:
    #        print "Null test", filename
    X_total = np.vstack([X_total, X_test])
    print X_total.shape
    print "Ready with test"
    print "Loading voxforge features"
    if VOX_FEATURES:
        if LIMIT is None:
            vox_limit = 4000
        else:
            vox_limit = LIMIT
        voxforge_features = load_vox_forge_files('/store/egor/voxforge', vox_limit, 'bow_voxfeatures.pcl', extract_mfcc_features )
        vox_features = np.vstack([f for (_,_,f) in voxforge_features if f is not None] )
        X_total = np.vstack([ X_total, vox_features ])
    else:
        voxforge_features = []

    print "Transformed everyting for dictionary learning", X_total.shape

    dictionary = learn_dictionary( sofia_path, X_total,'dictionary.pcl', DICTIONARY_SIZE )
    
    X = np.array([features for (_,_, features) in itertools.chain(X_train_transformed, voxforge_features) if features is not None ])
    #X = StandardScaler().fit_transform(X)
    y = np.array([lbl for (_,lbl, _) in itertools.chain(X_train_transformed, voxforge_features) if lbl is not None ])
    
    print "X = ", X.shape, "y=", y.shape
    #num_samples = len(all_train_data)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, y, train_size = 0.8)
    print "X_train = ", X_train.shape, "y_train =", Y_train.shape
    print "X_valid = ", X_valid.shape, "y_valid=", Y_valid.shape
    print "Training"
    bag_transformer = BagOfMFCCTransformer( dictionary, DICTIONARY_SIZE )
        
    X_train = bag_transformer.transform(X_train)
    X_valid = bag_transformer.transform(X_valid)
    print X_train.shape, Y_train.shape

    param_grid = {'C' : [10] }
    #gs = GridSearchCV( SVC(kernel = 'linear', probability = True), n_jobs = -1, verbose = 3,
    #                           param_grid=param_grid, refit = True, scoring='accuracy', cv=5)
    #clf = RandomForestClassifier(n_jobs = -1)
    
    #gs.fit(X_train, Y_train)

    logistic_params_grid = {'C' : [5], 'penalty' : ['l1','l2']}
    gbm_params_grid = { 'n_estimators' : [10,50,100,200,500] }
    svc_params_grid = {'C' : [1,10,100, 1000]}
    rf_params_grid = {'n_estimators' : [10,25,50,100,200]}

    param_grid = svc_params_grid
    #classifier = LogisticRegression()
    #gs = GridSearchCV( classifier, n_jobs = -1, verbose = 3,
    #                           param_grid= logistic_params_grid , refit = True, scoring='accuracy', cv=10)
    #clf = RandomForestClassifier(n_jobs = -1)
    #gs.fit(X_train.toarray(), Y_train)
    #clf = gs.best_estimator_
    #clf =  LinearSVC(penalty='l2', C = 10)
    clf = LogisticRegression(penalty='l2', C=10)
    clf.fit(X_train, Y_train)
    #clf = gs.best_estimator_

    print "Predictions"
    y_predicted = clf.predict( X_valid.toarray() )
    print classification_report( Y_valid, y_predicted )
    print confusion_matrix( Y_valid, y_predicted )
    total_X_transformed = bag_transformer.transform( X )
    print "Refitting"
    clf.fit( total_X_transformed.toarray(), y )
    joblib.dump( clf, 'linear_clf.pcl' )

    raw_predictions = open('./results/bow_predictions.txt', 'w')
    print "Starting predictions"
    with open('./results/bow_result.txt', 'w') as results_file:
        test_features = np.array([feature for (_,_,feature) in X_test_transformed ])
        #test_features = StandardScaler().fit_transform( test_features )
        transformed_features = bag_transformer.transform( test_features )
        print "Transformed output"
        print "predictioning"
        predictions = clf.predict_proba( transformed_features.toarray() )
        #predictions = clf.decision_function( transformed_features )
        print "Made preditions. Writing"
        for (t_idx,( full_path, dummy_lbl, features)) in enumerate(X_test_transformed):
            transformed = transformed_features[ t_idx ]
            predicted = [ (idx, proba) for (idx, proba) in enumerate( predictions[t_idx] )]
            sorted_by_freq = sorted(predicted, key = lambda x: x[1], reverse= True)
            for (idx, ( class_idx, freq)) in enumerate(sorted_by_freq):
                results_file.write("\"{},{},{}\",\n".format( filename_by_path[ full_path ], REVERSE_CLASSES[ class_idx ], idx + 1 ))
                raw_predictions.write( "{},{},{}\n".format( filename_by_path[ full_path ], REVERSE_CLASSES[class_idx], sorted_by_freq[ class_idx ][1] ) ) 
            results_file.flush()        
    raw_predictions.flush()
    raw_predictions.close()


