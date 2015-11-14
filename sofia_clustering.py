#-*-coding:utf-8-*-
from subprocess import call
import tempfile
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
import os
import numpy as np
__author__ = 'egor'


class SofiaClusterer(BaseEstimator):

    def __init__(self, k_means_path, num_clusters,mini_batch_size,iterations):
        self.cluster_centers_ = None
        self.num_clusters = num_clusters
        self.mini_batch_size = mini_batch_size
        self.iterations = iterations
        self.k_means_path = k_means_path
        self.clusters_path = None
        self.source_fn = '/home/egor/dic_source.txt'
        self.clusters_fn = '/home/egor/dic_clusters.txt'

    def fit(self, X, y = None):
        num_features = X.shape[1]
        if not os.path.exists( self.source_fn ):
            with open(self.source_fn, 'w') as f:
                for x in X:
                    sofia_string = ' '.join(["{0}:{1:.4f}".format( f_idx + 1, f_value )  for f_idx, f_value in enumerate(x) ])
                    total_string += "-1 {0}\n".format( sofia_string )
                f.write( total_string )
                f.flush()
                #run sofia
        print "Started clustering"
        command = '{} --k {} --init_type random --opt_type mini_batch_kmeans ' \
                          '--objective_after_init --objective_after_training --mini_batch_size {} --iterations {} ' \
                          '--training_file {} --model_out {} --dimensionality 10'.format(self.k_means_path,
                    self.num_clusters, self.mini_batch_size, self.iterations, self.source_fn, self.clusters_fn
                )
        os.system(command)
        print "Finished clustering"
        with open(self.clusters_fn,'r') as f:
            res = res.readlines()
            self.cluster_centers_ = np.array([[float(line.split()[feature_idx] )
                             for feature_idx in xrange(num_features) ] for line in res ])
                #print self.cluster_centers_.shape

    def transform(self, X):
        return self.predict(X)

    def predict(self, X):
        with tempfile.NamedTemporaryFile() as res:
            with tempfile.NamedTemporaryFile() as f:
                for x in X:
                    sofia_string = ' '.join(["{0}:{1:.5f}".format( f_idx, f_value )  for f_idx, f_value in enumerate(x) ])
                    total_string = "-1 {0}\n".format( sofia_string )
                    f.write( total_string )
                f.flush()
                command = '{} --model_in {} --test_file {} --objective_on_test --cluster_assignments_out {}'.format(
                    self.k_means_path, self.clusters_fn, f.name, res.name            )
                os.system(command)
                res = res.readlines()
                labels = [int(line.split()[0]) for line in res ]
                return np.array(labels)

if __name__ == "__main__":
    import numpy as np
    from sklearn.cluster import MeanShift, estimate_bandwidth
    from sklearn.datasets.samples_generator import make_blobs
    import sys
    ###############################################################################
    # Generate sample data
    centers = [[i,i]  for i in xrange(1000)]
    X, _ = make_blobs(n_samples=1000000, centers=centers, cluster_std=0.6)
    sofial_clusterer = SofiaClusterer( sys.argv[1] , 4, 5000, 10000)
    sofial_clusterer.fit(X)
    labels = sofial_clusterer.predict( X )
    ###############################################################################
    # Compute clustering with MeanShift

    # The following bandwidth can be automatically detected using
    #bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

    #ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    #ms.fit(X)
    labels = labels
    cluster_centers = sofial_clusterer.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)

    ###############################################################################
    # Plot result
    import matplotlib.pyplot as plt
    from itertools import cycle

    plt.figure(1)
    plt.clf()

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
