import itertools
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from features import extract_gmm_feature
from get_voxforge_data import load_vox_forge_files
from utils import REVERSE_CLASSES, get_all_test_data, \
    load_train_features, load_test_features, shuffle_in_unison_inplace
import numpy as np
from sklearn.grid_search import GridSearchCV
from sknn.mlp import Classifier, Layer
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
__author__ = 'egor'

VALIDATE = True
np.random.seed(100500)
VOX_FEATURES = True
LIMIT = 5000

print "Using Vox features", VOX_FEATURES
print "LIMIT", LIMIT

X_train_transformed = load_train_features('gmm_train_features.pcl', extract_gmm_feature, limit = LIMIT )
print "Loaded train"

test_data = get_all_test_data()
filename_by_path = {path : filename  for (path, filename) in test_data }
X_test_transformed = load_test_features('gmm_test_features.pcl', extract_gmm_feature, limit = LIMIT)
print "Loaded test"

if VOX_FEATURES:
    print "Loading voxforge features"
    if LIMIT is None:
        vox_limit = 4000
    else:
        vox_limit = LIMIT
    voxforge_features = load_vox_forge_files('/store/egor/voxforge', vox_limit, 'gmm_voxfeatures.pcl', extract_gmm_feature )
else:
    voxforge_features = []

#X = np.array([features for (_,_, features) in itertools.chain(X_train_transformed, voxforge_features)])
X = np.array([features for (_,_, features) in itertools.chain(X_train_transformed, voxforge_features) if features is not None ])
y = np.array([lbl for (_,lbl, _) in itertools.chain(X_train_transformed, voxforge_features) if lbl is not None ])
print "X = ", X.shape, "y=", y.shape
print "Loaded X", X.shape
#shuffle
X, y = shuffle_in_unison_inplace(X,y)
#X = StandardScaler().fit_transform(X)
#binarizer.fit( y )
#y = binarizer.transform( y )
print "Preprocessed data"
if VALIDATE:
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, y, train_size = 0.8)
    logistic_param_grid = {'C' : [1,10,100,1000,10000,100000], 'penalty' : ['l1','l2']}
    gbm_params_grid = { 'n_estimators' : [10,50,100,200,500,1000] }    
    svc_params_grid = {'C' : [1,10,100, 1000]}

    param_grid = svc_params_grid
    classifier = GradientBoostingClassifier()
    gs = GridSearchCV( classifier, n_jobs = -1, verbose = 3,
                               param_grid= gbm_params_grid , refit = True, scoring='accuracy', cv=10)
    #clf = RandomForestClassifier(n_jobs = -1)
    gs.fit(X_train, Y_train)
    clf = gs.best_estimator_
    #clf = LogisticRegression(C=5)
    #clf = LogisticRegression(C=5, penalty='l1' )
    #clf.fit(X_train, Y_train)
    #print clf
    y_predicted = clf.predict( X_valid )
    print classification_report( Y_valid, y_predicted )
    print confusion_matrix( Y_valid, y_predicted )
    clf.fit(X, y)
else:
    clf = LogisticRegression(C = 5)
    clf.fit(X, y)

print "Generating output"

raw_predictions = open('./results/gmm_predictions.txt', 'w')
with open('./results/gmm_result.txt', 'w') as results_file:
    test_features = np.vstack([feature for (_,_,feature) in X_test_transformed ])
    #test_features = StandardScaler().fit_transform( test_features )

    predictions = clf.predict_proba( test_features )
    for (t_idx,( full_path, dummy_lbl, features)) in enumerate(X_test_transformed):
        predicted = [ (idx, proba) for (idx, proba) in enumerate( predictions[t_idx] )]
        sorted_by_freq = sorted(predicted, key = lambda x: x[1], reverse= True)
        for (idx, ( class_idx, freq)) in enumerate(sorted_by_freq):
            results_file.write("\"{},{},{}\",\n".format( filename_by_path[ full_path ],
                                                         REVERSE_CLASSES[ class_idx ], idx + 1 ))
            raw_predictions.write( "{},{},{}\n".format( filename_by_path[ full_path ],
                                                        REVERSE_CLASSES[class_idx], sorted_by_freq[ class_idx ][1] ) )
        results_file.flush()
raw_predictions.flush()
raw_predictions.close()
