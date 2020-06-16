#!/usr/bin/python3

# Third party imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA as RandomizedPCA

# Local application imports
import BotDetection.pre_processing as pre_processing


def PCA(features_train, features_test):

    pca = RandomizedPCA(n_components=56, whiten=True, random_state=0).fit(features_train)

    features_train = pca.transform(features_train)
    features_test = pca.transform(features_test)

    return features_train, features_test, pca


def distortionVectorizer(text_Train, text_Test):
    vectorizer = TfidfVectorizer(analyzer='char', sublinear_tf=True, min_df=5, ngram_range=(2, 8), max_features=1000)
    text_Train = (vectorizer.fit_transform(pre_processing.list_distortion(text_Train)))
    text_Test = (vectorizer.transform(pre_processing.list_distortion(text_Test)))

    return text_Train, text_Test, vectorizer