"""
Wrapper for offline clustering methods that do not take into
account temporal aspects of data and online clustering methods
that update and/or predict new data as it comes in. Framework
supports custom text representations (e.g. Continuous Bag of
Words) but will default to tfidf if none are provided.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from hdbscan import HDBSCAN
from nltk.corpus import stopwords
from scipy.sparse import issparse, vstack
from sklearn.cluster import *
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

nltk_stopwords = stopwords.words('english')


class Cluster:
    """ Clustering methods for text. Be cautious of datasize; in cases
    of large data, KMeans may be the only efficient choice.

    Accepts custom matrices

    Full analysis of methods can be found at:
    http://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html

    Usage:
        >> with open('../data/cleaned_text.txt', 'r', encoding='utf8') as f:
               text = f.readlines()
        >> clustering = Cluster(text)
        >> results = clustering('hdbscan', matrix=None, reduce_dim=None,
                                visualize=True, top_terms=False,
                                min_cluster_size=10)
        >> print(results)
    """
    def __init__(self, text):
        """
        Args:
            text: strings to be clustered (list of strings)
        """
        self.text = list(set(text))

    def __call__(self, method, vectorizer=None,
                         reduce_dim=None, viz=False,
                         *args, **kwargs):
        """
        Args:
            method: algorithm to use to cluster data (str)
            vectorizer: initialized method to convert text to np array;
                        assumes __call__ vectorizes the text (Class, optional)
            reduce_dim: reduce dim of representation matrix (int, optional)
            visualize: visualize clusters in 3D (bool, optional)
            *args, **kwargs: see specified method function
        """

        # Make sure method is valid
        assert method in ['hdbscan', 'dbscan', 'spectral',
                          'kmeans', 'affinity_prop', 'agglomerative',
                          'mean_shift', 'birch'], 'Invalid method chosen.'

        if not hasattr(self, 'vectorizer'):
            if vectorizer is None:
                self._init_tfidf()
            else:
                self.vectorizer = vectorizer
                self.matrix = self.vectorizer(self.text)

        # Reduce dimensionality using latent semantic analysis (makes faster)
        if reduce_dim is not None:
            self.matrix = self._pca(reduce_dim, self.matrix)

        # Cache current method
        method = eval('self.' + method)
        self.algorithm = method(*args, **kwargs)
        self.results = self._organize(self.algorithm.labels_)

        # For plotting
        self.viz_matrix = self.matrix

        # Visualize clustering outputs if applicable
        if viz:
            _ = self.viz3D()
            _ = self.top_terms()

        return self.results

    def hdbscan(self, min_cluster_size=10, prediction_data=False):
        """ DBSCAN but allows for varying density clusters and no longer
        requires epsilon parameter, which is difficult to tune.
        http://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html
        Scales slightly worse than DBSCAN, but with a more intuitive parameter.
        """
        hdbscan = HDBSCAN(min_cluster_size=min_cluster_size,
                            prediction_data=prediction_data)
        if prediction_data:
            return hdbscan.fit(self._safe_dense(self.matrix))
        else:
            return hdbscan.fit(self.matrix)

    def dbscan(self, eps=0.50):
        """ Density-based algorithm that clusters points in dense areas and
        distances points in sparse areas. Stable, semi-fast, non-global.
        Scales very well with n_samples, decently with n_clusters (not tunable)
        """
        dbscan = DBSCAN(eps=eps)
        return dbscan.fit(self.matrix)

    def kmeans(self, n_clusters=10, n_init=5, batch_size=5000):
        """ Partition dataset into n_cluster global chunks by minimizing
        intra-partition distances. Expect quick results, but with noise.
        Scales exceptionally well with n_samples, decently with n_clusters.
        """
        kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                                 init='k-means++',
                                 n_init=n_init,
                                 batch_size=batch_size)
        return kmeans.fit(self.matrix)

    def birch(self, n_clusters=10):
        """ Partitions dataset into n_cluster global chunks by repeatedly
        merging subclusters of a CF tree. Birch does not scale very well to high
        dimensional data. If many subclusters are desired, set n_clusters=None.
        Scales well with n_samples, well with n_clusters.
        """
        birch = Birch(n_clusters=n_clusters)
        return birch.fit(self.matrix)

    def agglomerative(self, n_clusters=10, linkage='ward'):
        """ Iteratively clusters dataset semi-globally by starting with each
        point in its own cluster and then using some criterion to choose another
        cluster to merge that cluster with another cluster.
        Scales well with n_samples, decently with n_clusters.
        """
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters,
                                                linkage=linkage)
        return agglomerative.fit(self._safe_dense(self.matrix))

    def spectral(self, n_clusters=5):
        """ Partitions dataset semi-globally by inducing a graph based on the
        distances between points and trying to learn a manifold, and then
        running a standard clustering algorithm (e.g. KMeans) on this manifold.
        Scales decently with n_samples, poorly with n_clusters.
        """
        spectral = SpectralClustering(n_clusters=n_clusters)
        return spectral.fit(self.matrix)

    def affinity_prop(self, damping=0.50):
        """ Partitions dataset globally using a graph based approach to let
        points ‘vote’ on their preferred ‘exemplar’.
        Does not scale well with n_samples. Not recommended to use with text.
        """
        affinity_prop = AffinityPropagation(damping=damping)
        return affinity_prop.fit(self._safe_dense(self.matrix))

    def mean_shift(self, cluster_all=False):
        """ Centroid-based, global method that assumes there exists some
        probability density function from which the data is drawn, and tries to
        place centroids of clusters  at the maxima of that density function.
        Unstable, but conservative.
        Does not scale well with n_samples. Not recommended to use with text.
        """
        mean_shift = MeanShift(cluster_all=False)
        return mean_shift.fit(self._safe_dense(self.matrix))

    def _init_tfidf(self, max_features=30000, analyzer='word',
                    stopwords=nltk_stopwords, token_pattern=r"(?u)\b\w+\b"):
        """ Default representation for data is sparse tfidf vectors

        Args:
            max_features: top N vocabulary to consider (int)
            analyzer: 'word' or 'char', level at which to segment text (str)
            stopwords: words to remove from consideration, default nltk (list)
        """
        # Initialize and fit tfidf vectors
        self.vectorizer = TfidfVectorizer(max_features=max_features,
                                             stop_words=stopwords,
                                             analyzer=analyzer,
                                             token_pattern=token_pattern)
        self.matrix = self.vectorizer.fit_transform(self.text)

        # Get top max_features vocabulary
        self.terms = self.vectorizer.get_feature_names()

        # For letting user know if tfidf has been initialized
        self.using_tfidf = True

    def viz2D(self, matrix=None,
                plot_kwds={'alpha':0.30, 's':40, 'linewidths':0}):
        """ Visualize clusters in 2D """
        # Run PCA over the data so we can plot
        matrix2D = self._pca(n=2, matrix=self.viz_matrix)

        # Get labels
        labels = np.unique(self.results['labels'])

        # Assign a color to each label
        palette = sns.color_palette('deep', max(labels)+1)
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]

        # Plot the data
        plt.close()
        fig = plt.figure(figsize=(10,6))
        plt.scatter(matrix2D.T[0],
                    matrix2D.T[1],
                    c=colors,
                    **plot_kwds)
        frame = plt.gca()

        # Turn off axes, since they are arbitrary
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)

        # Add a title
        alg_name = str(self.algorithm.__class__.__name__)
        plt.title('{0} clusters found by {1}'.format(len(labels),
                                                     alg_name),
                  fontsize=20)
        plt.tight_layout()
        plt.show()
        return fig

    def viz3D(self, matrix=None):
        """ Visualize clusters in 3D """
        # Run PCA over the data
        matrix3D = self._pca(n=3, matrix=self.viz_matrix)

        # Extract labels from results
        labels = self.results['labels']

        # Assign colors
        palette = sns.color_palette('deep', int(max(labels)+1))
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]

        # Plot the data
        plt.close()
        fig = plt.figure(figsize=(10,6))
        ax = plt.axes(projection='3d')
        ax.scatter(matrix3D.T[0],
                   matrix3D.T[1],
                   matrix3D.T[2],
                   c=colors)

        # Add a title
        alg_name = str(self.algorithm.__class__.__name__)
        plt.title('{0} Clusters | {1} Items | {2}'.format(len(set(labels)),
                                                            matrix3D.shape[0],
                                                            alg_name),
                  fontsize=20)

        # Turn off arbitrary axis tick labels
        plt.tick_params(axis='both', left=False, top=False, right=False,
                        bottom=False, labelleft=False, labeltop=False,
                        labelright=False, labelbottom=False)
        plt.tight_layout()
        plt.show()
        return fig

    def top_terms(self, topx=10):
        """ Print out top terms per cluster. """
        if self.using_tfidf != True:
            print('For use with non-tfidf vectorizers,try sklearn NearestNeighbors\
            (although NN performs poorly with high dimensional inputs.')
            return None

        # Get labels, sort text IDs by cluster
        labels = self.results['labels']
        cluster_idx = {clust_id: np.where(labels == clust_id)[0]
                       for clust_id in set(labels)}

        # Get centers, stack into array
        centroids = np.vstack([self.viz_matrix[indexes].mean(axis=0)
                                for key, indexes in cluster_idx.items()])

        # Compute closeness of each term representation to each centroid
        order_centroids = np.array(centroids).argsort()[:, ::-1]

        # Organize terms into a dictionary
        cluster_terms = {clust_id: [self.terms[ind]
                                    for ind in order_centroids[idx, :topx]]
                        for idx, clust_id in enumerate(cluster_idx.keys())}

        # Print results
        print("Top terms per cluster:")
        for clust_id, terms in cluster_terms.items():
            words = ' | '.join(terms)
            print("Cluster {0} ({1} items): {2}".format(clust_id,
                                                        len(cluster_idx[clust_id]),
                                                        words))

        return cluster_terms

    def item_counts(self):
        """ Print number of counts in each cluster """
        for key, vals in self.results.items():
            if key == 'labels':
                continue
            print('Cluster {0}: {1} items'.format(key, len(vals)))

    def _organize(self, labels):
        """ Organize text from clusters into a dictionary """
        # Organize text into respective clusters
        cluster_idx = {clust_id: np.where(labels == clust_id)[0]
                       for clust_id in set(labels)}

        # Put results in a dictionary; key is cluster idx values are text
        results = {clust_id: [self.text[idx] for idx in cluster_idx[clust_id]]
                    for clust_id in cluster_idx.keys()}
        results['labels'] = list(labels)

        return results

    def _pca(self, n, matrix):
        """ Perform PCA on the data """
        return TruncatedSVD(n_components=n).fit_transform(matrix)

    def _safe_dense(self, matrix):
        """ Some algorithms don't accept sparse input; for these, make
        sure the input matrix is dense. """
        if issparse(matrix):
            return matrix.todense()
        else:
            return matrix


class OnlineCluster(Cluster):
    """ Online (stream) clustering of textual data. Check each method
    to determine if the model is updating or ad-hoc predicting. These are not
    'true' online methods as they preserve all seen data, as opposed to letting
    data points and clusters fade, merge, etc. over time.

    Usage:
        To initialize:
        >> with open('../data/cleaned_text.txt', 'r', encoding='utf8') as f:
               text = f.readlines()
        >> online = OnlineCluster(method='kmeans', text, visualize=True)

        To predict and update parameters if applicable:
        >> new_text = text[-10:]
        >> online.predict(new_text)
    """
    def __init__(self, text, method, *args, **kwargs):
        """
        Args:
            text: strings to be clustered (list of strings)
            method: algorithm to use to cluster (string)
            *args, **kwargs (optional):
                vectorizer: text representation. Defaults tfidf (array, optional)
                reduce_dim: reduce dim of representation matrix (int, optional)
                visualize: visualize clusters in 3D (bool, optional)
        """
        # Only accept valid arguments
        assert method in ['kmeans', 'birch', 'hdbscan',
                          'dbscan', 'mean_shift'], \
                'Method incompatible with online clustering.'

        # Initialize inherited class
        super().__init__(text)

        # Get initial results
        self.results = self.__call__(method=method, *args,**kwargs)

        # Save args, set method
        self.__dict__.update(locals())
        self.method = eval('self._' + method)

    def predict(self, new_text):
        """ 'Predict' a new example based on cluster centroids and update params
        if applicable (kmeans, birch). If a custom (non-tfidf) text representation
        is being used, class assumes new_text is already in vectorized form.

        Args:
            new_text: list of strings to predict
        """
        # Predict
        assert type(new_text) == list, 'Input should be list of strings.'
        self.text = list(set(self.text + new_text))
        new_matrix = self._transform(new_text)
        output_labels = self.method(new_matrix)

        # Update attribute for results, plotting
        self._update_results(output_labels)
        self.viz_matrix = vstack([self.viz_matrix, new_matrix])
        return output_labels

    def _kmeans(self, new_matrix):
        """ Updates parameters and predicts """
        self.algorithm = self.algorithm.partial_fit(new_matrix)
        return self.algorithm.predict(new_matrix)

    def _birch(self, new_matrix):
        """ Updates parameters and predicts """
        self.algorithm = self.algorithm.partial_fit(new_matrix)
        return self.algorithm.predict(new_matrix)

    def _hdbscan(self, new_matrix):
        """ Prediction only, HDBSCAN requires training to be done on dense
        matrices for prediction to work properly. This makes training
        inefficient, though. """
        try:
            labels, _ = approximate_predict(self.algorithm,
                                            self._safe_dense(new_matrix))
        except AttributeError:
            try:
                self.algorithm.generate_prediction_data()
                labels, _ = approximate_predict(self.algorithm,
                                                self._safe_dense(new_matrix))
            except ValueError:
                print('Must (inefficiently) re-train with prediction_data=True')
        return labels

    def _dbscan(self, new_matrix):
        """ Prediction only """
        # Extract labels
        labels = self.algorithm.labels_

        # Result is noise by default
        output = np.ones(shape=new_matrix.shape[0], dtype=int)*-1

        # Iterate all input samples for a label
        for idx, row in enumerate(new_matrix):

            # Find a core sample closer than EPS
            for i, row in enumerate(self.algorithm.components_):

                # If it's below the threshold of the dbscan model
                if cosine(row, x_core) < self.algorithm.eps:

                    # Assign label of x_core to the input sample
                    output[idx] = labels[self.algorithm.core_sample_indices_[i]]
                    break

        return output

    def _mean_shift(self, new_matrix):
        """ Prediction only, not efficient """
        return self.algorithm.predict(new_matrix)

    def _transform(self, new_text):
        """ Transform text to tfidf representation. Assumes already vectorized
        if tfidf matrix has not been initialized. """
        if self.using_tfidf:
            return self.vectorizer.transform(new_text)
        else:
            return self.vectorizer(new_text)
        return new_matrix

    def _update_results(self, labels):
        """ Update running dictionary """
        new_results = self._organize(labels)
        for key in self.results.keys():
            try:
                self.results[key] += new_results[key]
            except KeyError:
                continue
