#!/usr/bin/python3
from nltk import TweetTokenizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


def TFIDF(feature_train, feature_test):
    vectorizer = TfidfVectorizer(tokenizer=TweetTokenizer(reduce_len=True).tokenize, analyzer='word',
                                 lowercase=True,
                                 stop_words='english', min_df=4, sublinear_tf=True, ngram_range=(1, 5),
                                 max_features=90000)

    feature_train = (vectorizer.fit_transform(feature_train)).toarray()
    feature_test = (vectorizer.transform(feature_test)).toarray()
    return feature_train, feature_test, vectorizer


def truncated_SVD(feature_train):
    svd = TruncatedSVD(n_components=11, n_iter=22, random_state=0)
    feature_train = svd.fit_transform(feature_train)
    return svd, feature_train
