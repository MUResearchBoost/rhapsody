from time import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from lda.toolfunction import print_top_words


class Clustering_algorithm:
    option = ['Lda', 'Nmf']

    def lda_data(self, data_samples, n_samples, n_features, n_components, n_top_words):
        # Use tf (raw term count) features for LDA.
        print("Extracting tf features for LDA...")
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=n_features,
                                        stop_words='english')
        t0 = time()
        tf = tf_vectorizer.fit_transform(data_samples)
        print("done in %0.3fs." % (time() - t0))
        print()

        print("Fitting LDA models with tf features, "
              "n_samples=%d and n_features=%d..."
              % (n_samples, n_features))
        lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)
        t0 = time()
        lda.fit(tf)
        print("done in %0.3fs." % (time() - t0))

        print("\nTopics in lda model (Frobenius norm):")
        tf_feature_names = tf_vectorizer.get_feature_names()
        keywords = print_top_words(lda, tf_feature_names, n_top_words)

        topic_distribution = lda.transform(tf)
        topic_distribution_filtered = np.where(topic_distribution >= 0.2, topic_distribution, 0)
        topic_distribution_filtered = np.where(topic_distribution_filtered == 0, topic_distribution_filtered, 1)
        # rows = np.arange(0, 8000, 1)
        topics = np.where(topic_distribution_filtered == np.max(topic_distribution_filtered, axis=0))
        # lda_distribution = renewed_newsgroup(labels)
        # print(lda_distribution)

        return tf, topics, keywords









