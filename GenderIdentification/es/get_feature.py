#!/usr/bin/python3

def TFIDF(feature_train, feature_test):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.tokenize import TweetTokenizer

    vectorizer = TfidfVectorizer(max_features=50000, tokenizer=TweetTokenizer(reduce_len=True).tokenize, analyzer='word', lowercase=True,
                                 stop_words='english', min_df=8, sublinear_tf=True, ngram_range=(1, 9), use_idf=False)

    feature_train = (vectorizer.fit_transform(feature_train)).toarray()
    if feature_test is not None:
        feature_test = (vectorizer.transform(feature_test)).toarray()

    return feature_train, feature_test, vectorizer

def LSA(feature_train, feature_test):
    from sklearn.decomposition import TruncatedSVD

    lsa = TruncatedSVD(n_components=11, n_iter=22, random_state=0)  # .fit(feature_train)

    feature_train = lsa.fit_transform(feature_train)
    feature_test = lsa.transform(feature_test)
    return feature_train, feature_test, lsa