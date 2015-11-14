#-*-coding:utf-8-*-
__author__ = 'egor'
import os
from path import Path
from multiprocessing import Pool
import cPickle
import time

def saver( data):
    filename, save_object = data
    with open(filename+'.pcl', 'wb') as f:
        pickler = cPickle.Pickler(f, cPickle.HIGHEST_PROTOCOL)
        pickler.fast = 1
        pickler.dump(save_object)


def loader( filename ):
    with open( filename, 'rb' ) as f:
        unpickler = cPickle.Unpickler(f)
        obj = unpickler.load()
        return obj

class FeatureStorage:

    def __init__(self, name, base_dir):
        """

        :param name:
        :param base_dir:
        :return:
        """
        self.directory_for_storage = os.path.abspath(os.path.join( base_dir, name ))


    def exists(self):
        return os.path.exists( self.directory_for_storage )

    def save(self, features):
        """

        :param features:  (filename, lbl, numpy array)
        :return:
        """
        #save parallel many files
        if not self.exists():
            os.makedirs( self.directory_for_storage )
        start_time = time.time()
        jobs = [(os.path.join( self.directory_for_storage, os.path.basename(filename) ), (filename,lbl, features) )
                for (filename,lbl, features) in features ]
        pool = Pool()
        pool.map( saver, jobs )
        pool.close()
        pool.terminate()
        took = time.time() - start_time
        print "Saving took", took


    def load( self ):
        start_time = time.time()
        train_dir = Path( self.directory_for_storage )
        filenames = []
        for f in train_dir.walkfiles('*.pcl'):
            filenames.append( str(f) )
        pool = Pool()
        objects = pool.map( loader, filenames )
        took = time.time() - start_time
        print "Loading took", took
        return objects