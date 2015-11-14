#-*-coding:utf-8-*-
import joblib
import sys, os
from bag_of_words import BagOfMFCCTransformer
from features import extract_mfcc_features
from utils import REVERSE_CLASSES
import numpy as np
DICTIONARY_SIZE = 600


if __name__ == "__main__":
    #1.load bag-of-words dictionary
    clusterer = joblib.load( 'dictionary.pcl' )
    clusterer.verbose = 0
    #2.load model
    clf = joblib.load( 'linear_clf.pcl' )        
    #3.bag of words transformer
    bag_transformer = BagOfMFCCTransformer( clusterer, DICTIONARY_SIZE )

    
    #4. process input file

    input_file = sys.argv[1]
    print "Processing {}".format( input_file )
    _,_, features = extract_mfcc_features( (input_file,None) )
    
    with open('1.csv', 'w') as results_file:
        transformed_features = bag_transformer.transform( np.array([features]) )
        print "Transformed output"
        print "predictioning"
        predictions = clf.predict_proba( transformed_features.toarray() )[0]
        print "Made preditions. Writing to 1.csv"
        predicted = [ (idx, proba) for (idx, proba) in enumerate( predictions )]
        sorted_by_freq = sorted(predicted, key = lambda x: x[1], reverse= True)
        for (idx, ( class_idx, freq)) in enumerate(sorted_by_freq):
            results_file.write("\"{},{},{}\",\n".format( input_file, REVERSE_CLASSES[ class_idx ], idx + 1 ))
        results_file.flush()
