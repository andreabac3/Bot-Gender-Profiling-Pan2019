#!/usr/bin/python3
# coding=utf-8


# Standard library imports
import os
import re
import numpy as np
import numpy.linalg as LA

# Third party imports
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer

import xml.etree.ElementTree as ET
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def list_distortion(texts):
    res = [distortion(text) for text in texts]
    return res


def distortion(string):
    # use not is_char per avere *Danger* b laisser m* It*s dry*
    # string = 'la miò stringà ?? con. ZZ ` [] \_ed aa 9:A@.'

    def is_char(char):
        # Remove all char that is not [a-z,A-Z,0-9]
        # Check if is a char in [a-z,A-Z,0-9]
        return 48 < ord(char) < 123 and not 90 < ord(char) < 97 and not 57 < ord(char) < 65

    # https://ascii.cl/
    string = string.replace('\n', " ").replace("  ", "")
    s = ''.join(['*' if char != ' ' and is_char(char) else char for char in string])
    return s



def processXML3(filename, nome, LANGUAGE):
    stemmer = SnowballStemmer(LANGUAGE)
    useSentiment = True

    analyzer = SentimentIntensityAnalyzer()
    useSentiment = True

    cntRetweet = 0
    cnt = 0
    cntWebsite = 0
    cntLun = 0
    cntLun2 = 0
    numWebsite = 0
    preFix = "<document><![CDATA["
    postFix = "]]></document>"
    numEmoji = 0
    virgole = 0
    punti = 0
    puntivirgola = 0
    root = ET.parse(filename).getroot()
    text = ''
    cntEnter = 0
    cntStopWords = 0
    dictionaryWord = {}
    listMessage = []
    i = 0
    numHashtagTotali = 0
    hasHashtag = 0
    isBot = 0
    finishWithLink = 0
    fakeNewsPoint = 0
    negative, positive, neutral, compound = 0, 0, 0, 0
    bag_of_words_bot = ["bot", "b0t", "cannabis", "tweet me", "mishear", "follow me", "updates every", "gorilla",
                        "yes_ofc", "forget , expos", "kill", "clit", "bbb", "butt", "fuck", "XXX", "sex", "truthe",
                        "fake", "anony", "free", "virus", "funky", "RNA", "kuck", "jargon"]
    for lineXML in root.iter('document'):
        x = lineXML.text



        vs = analyzer.polarity_scores(x)  # sentiment english
        positive += vs['pos']
        neutral += vs['neu']
        negative += vs['neg']
        compound += vs['compound']

        numWebsite += x.count("https://")
        numWebsite += x.count("http://")
        numHashtagTotali += x.count("#")
        if "#" in x:
            hasHashtag += 1
        numEmoji += len(re.findall(r'[^@\w\s,]', x))
        virgole += x.count(", ")
        puntivirgola += x.count("; ")
        punti += x.count(". ")
        cnt += 1
        i += 1
        cntLun += len(x)
        cntTag, cntTag3, cntTAG2 = 0, 0, 0
        cntEnter = x.count("\n")
        #x = p.tokenize(x)
        y = lineXML.text.split()  # word list of a Tweet
        #y = x.split()  # word list of a Tweet

        testo = ""
        iter = False
        for word in y:

            # sostituisci:  sito e numeri
            wordStem = stemmer.stem(word)
            if word in bag_of_words_bot or wordStem in bag_of_words_bot and not iter:
                isBot += 1
                iter = True
                #print(filename)
            text += (wordStem + "\n")
            testo += wordStem + " "
            dictionaryWord.update({wordStem: dictionaryWord[wordStem] + 1 if wordStem in dictionaryWord else 1})
        # listMessage.append(re.sub(r'rt @(\w+)', "::RETWEET::", re.sub(r'[^@\w\s,]', " ::EMOJI:: ", testo)))
        listMessage.append(re.sub(r'[^@\w\s,]', " ::EMOJI:: ", testo))  # replace of every emoji icon with tag
        if "RT @" in x:
            cntRetweet += 1
            cntLun2 += len(x)
        if "https://" in x or "http://" in x or "www." in x or ".com" in x:
            cntWebsite += 1
        if "RT @" not in x and "@" in x:
            cntTag += 1
            cntTAG2 += x.count("@")
        if "@" in x:
            cntTag3 += x.count("@")

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(listMessage)

    from sklearn.metrics.pairwise import cosine_similarity
    dict = {"fakeNews": fakeNewsPoint/100, "positive": positive / 100,
            "negative": negative + positive / 100, "neutral": neutral / 100, "compound": compound / 100, "isBot": isBot,
            "hasHashtag": hasHashtag/100, "numHashtagTotali": numHashtagTotali/100,
            "similarity": cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)[0, 1], "author": nome, "cntTag3": cntTag3,
            "cntTag": cntTag, "cntTAG2": cntTAG2,
            "cntStopWords": cntStopWords, "punti": punti,
            "puntivirgola": puntivirgola,
            "virgola": virgole, "RT": cntRetweet, "len": cntLun, "web": cntWebsite,
            "lenMedia": cntLun / cnt if cnt > 0 else 0,
            "lenRT": cntLun2, "lenMediaRT": cntLun2 / cntRetweet if cntRetweet > 0 else 0, "numEmoji": numEmoji,
            "mediaEmoji": numEmoji / cnt if cnt > 0 else 0, "cnt": cnt, "text": text,
            "cntEnter": cntEnter, "countDiffword": len(set(text)), "cntTag4":cntTag3/100}
    return dict

def processXML(filename, nome, LANGUAGE):
    stemmer = SnowballStemmer(LANGUAGE)
    analyzer = SentimentIntensityAnalyzer()
    cntRetweet = 0
    cnt = 0
    cntWebsite = 0
    cntLun = 0
    cntLun2 = 0
    numWebsite = 0
    preFix = "<document><![CDATA["
    postFix = "]]></document>"
    numEmoji = 0
    virgole = 0
    punti = 0
    puntivirgola = 0
    root = ET.parse(filename).getroot()
    text = ''
    cntStopWords = 0
    dictionaryWord = {}
    listMessage = []
    i = 0
    numHashtagTotali = 0
    hasHashtag = 0
    isBot = 0
    finishWithLink = 0
    fakeNewsPoint = 0
    tweets = []
    negative, positive, neutral, compound = 0, 0, 0, 0

    for lineXML in root.iter('document'):
        x = lineXML.text

        vs = analyzer.polarity_scores(x)  # sentiment english
        positive += vs['pos']
        neutral += vs['neu']
        negative += vs['neg']
        compound += vs['compound']

        numWebsite += x.count("https://")
        numWebsite += x.count("http://")
        numHashtagTotali += x.count("#")
        if "#" in x:
            hasHashtag += 1
        numEmoji += len(re.findall(r'[^@\w\s,]', x))
        virgole += x.count(", ")
        puntivirgola += x.count("; ")
        punti += x.count(". ")
        cnt += 1
        i += 1
        cntLun += len(x)
        cntTag, cntTag3, cntTAG2 = 0, 0, 0
        #x = p.tokenize(x)
        y = lineXML.text
        tweets.append(y)
        #y = x.split()  # word list of a Tweet
        testo = ""
        for word in y.split() :
            # sostituisci:  sito e numeri
            wordStem = stemmer.stem(word)
            text += (wordStem + "\n")
            testo += wordStem + " "
            dictionaryWord.update({wordStem: dictionaryWord[wordStem] + 1 if wordStem in dictionaryWord else 1})
        # listMessage.append(re.sub(r'rt @(\w+)', "::RETWEET::", re.sub(r'[^@\w\s,]', " ::EMOJI:: ", testo)))
        listMessage.append(re.sub(r'[^@\w\s,]', " ::EMOJI:: ", testo))  # replace of every emoji icon with tag
        if "RT @" in x:
            cntRetweet += 1
            cntLun2 += len(x)
        if "https://" in x or "http://" in x or "www." in x or ".com" in x:
            cntWebsite += 1
        if "RT @" not in x and "@" in x:
            cntTag += 1
            cntTAG2 += x.count("@")
        if "@" in x:
            cntTag3 += x.count("@")

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(listMessage)
    cosineSimilarity = np.mean(np.sum(cosine_similarity(tfidf_matrix), axis=0))

    dict = {"fakeNews": fakeNewsPoint/100, "positive": positive / 100,
            "negative": negative + positive / 100, "neutral": neutral / 100, "compound": compound / 100, "isBot": isBot,
            "hasHashtag": hasHashtag/100, "numHashtagTotali": numHashtagTotali/100,
            "similarity": cosineSimilarity, "author": nome, "cntTag3": cntTag3,
            "cntTag": cntTag, "cntTAG2": cntTAG2,
            "cntStopWords": cntStopWords, "punti": punti,
            "puntivirgola": puntivirgola,
            "virgola": virgole, "RT": cntRetweet, "len": cntLun, "web": cntWebsite,
            "lenMedia": cntLun / cnt if cnt > 0 else 0,
            "lenRT": cntLun2, "lenMediaRT": cntLun2 / cntRetweet if cntRetweet > 0 else 0, "numEmoji": numEmoji,
            "mediaEmoji": numEmoji / cnt if cnt > 0 else 0, "cnt": cnt, "text": text,
            "countDiffword": len(set(text)), "cntTag4":cntTag3/100, "tweets":tweets}
    return dict


def processXML2(filename, nome, LANGUAGE):
    stemmer = SnowballStemmer(LANGUAGE)
    analyzer = SentimentIntensityAnalyzer()

    cntRetweet = 0
    cnt = 0
    cntWebsite = 0
    cntLun = 0
    cntLun2 = 0
    numWebsite = 0
    preFix = "<document><![CDATA["
    postFix = "]]></document>"
    numEmoji = 0
    virgole = 0
    puntivirgola = 0
    root = ET.parse(filename).getroot()
    text = ''
    dictionaryWord = {}
    listMessage = []
    i = 0
    numHashtagTotali = 0
    hasHashtag = 0
    negative, positive, compound = 0, 0, 0

    for lineXML in root.iter('document'):
        x = lineXML.text

        vs = analyzer.polarity_scores(x)  # sentiment english
        positive += vs['pos']
        negative += vs['neg']
        compound += vs['compound']

        numWebsite += x.count("https://")
        numWebsite += x.count("http://")
        numHashtagTotali += x.count("#")
        if "#" in x:
            hasHashtag += 1
        numEmoji += len(re.findall(r'[^@\w\s,]', x))
        virgole += x.count(", ")
        cnt += 1
        i += 1
        cntLun += len(x)
        cntTag, cntTag3, cntTAG2 = 0, 0, 0
        #y = lineXML.text.split()  # word list of a Tweet\
        tknzr = TweetTokenizer(reduce_len=True)
        y = tknzr.tokenize(lineXML.text)
        testo = ""
        for word in y:
            # sostituisci:  sito e numeri
            wordStem = word # stemmer.stem(word)
            # print(filename)
            text += (wordStem + "\n")
            testo += wordStem + " "
            dictionaryWord.update({wordStem: dictionaryWord[wordStem] + 1 if wordStem in dictionaryWord else 1})
        listMessage.append(re.sub(r'[^@\w\s,]', " ::EMOJI:: ", testo))  # replace of every emoji icon with tag
        if "RT @" in x:
            cntRetweet += 1
            cntLun2 += len(x)
        if "https://" in x or "http://" in x or "www." in x or ".com" in x:
            cntWebsite += 1
        if "RT @" not in x and "@" in x:
            cntTag += 1
            cntTAG2 += x.count("@")
        if "@" in x:
            cntTag3 += x.count("@")
    tfidf_vectorizer = TfidfVectorizer()

    cosine_value = 0
    cx = lambda a, b: round(np.inner(a, b) / (LA.norm(a) * LA.norm(b)), 3)

    for i in range(len(listMessage)-1):
        test_tweet = listMessage[i]
        del listMessage[i]
        train = tfidf_vectorizer.fit_transform(listMessage)
        test = tfidf_vectorizer.transform(test_tweet)
        cosine_value += cx(train, test)
        listMessage.insert(test, i)
    cosine_value //= 100
    '''
    ["label", "similarity", "len", "web", "lenMedia", "lenRT", "lenMediaRT", "mediaEmoji", "puntivirgola",
     "cntTag3", "cntTag", "cntTAG2", "compound", "negative", "hasHashtag"]
     '''
    '''
    ["label", "similarity", "len", "web", "lenRT", "lenMediaRT", "puntivirgola", "compound", "negative"] 
    '''
    # triu the values under the diagonal are only zeros.
    #cosineSimilarity = np.mean(np.sum(a=np.triu(cosine_similarity(tfidf_matrix), k=1), axis=1))
    # cosineSimilarity = np.mean(np.sum(cosine_similarity(tfidf_matrix), axis=1))
    '''
    print(np.mean(np.sum(cosine_similarity(tfidf_matrix), axis=0)))
    print("\n [0] -> ", (cosine_similarity(tfidf_matrix)[0]))
    print("\n GENERICA ->  ", (cosine_similarity(tfidf_matrix)))
    print("\n MANUAL SUM -> ", sum(cosine_similarity(tfidf_matrix)[0]))
    exit(0)
    '''
    #cosineSimilarity = np.mean(np.sum(cosine_similarity(tfidf_matrix), axis=0))

    dict = {
        "negative": negative + positive / cnt, "compound": compound / cnt,
        "hasHashtag": hasHashtag / cnt,
        "similarity": cosine_value, "author": nome, "cntTag3": cntTag3,
        "cntTag": cntTag, "cntTAG2": cntTAG2,
        "puntivirgola": puntivirgola, "len": cntLun, "web": cntWebsite,
        "lenMedia": cntLun / cnt if cnt > 0 else 0,
        "lenRT": cntLun2, "lenMediaRT": cntLun2 / cntRetweet if cntRetweet > 0 else 0, "numEmoji": numEmoji,
        "mediaEmoji": numEmoji / cnt if cnt > 0 else 0, "cnt": cnt, "text": "\n".join(listMessage)
    }
    return dict




def buildDataset(base_folder, filename, LANGUAGE):
    namefolder = "es"
    if LANGUAGE == "english":
        namefolder = "en"
    namefolder = base_folder + os.sep + namefolder + os.sep
    xml_directory = namefolder
    dataSet = {}
    with open(namefolder + filename, "r") as file:
        for line in file:
            truth_column = line.split(":::")
            author = truth_column[0]
            dict = processXML3(xml_directory + author + ".xml", author, LANGUAGE)
            isBot = truth_column[1]
            if isBot == "bot":
                dict.update({"label": 0})
            elif isBot == "human":
                dict.update({"label": 1})
            dataSet.update({author: dict})
    return dataSet


def targetFeatureSplit(data):
    """
        given a numpy array like the one returned from
        featureFormat, separate out the first feature
        and put it into its own list (this should be the
        quantity you want to predict)

        return targets and features as separate lists

        (sklearn can generally handle both lists and numpy arrays as
        input formats when training/predicting)
    """
    target = []
    features = []
    for item in data:
        target.append(item[0])
        features.append(item[1:])

    return target, features


def featureFormat(dictionary, features, remove_NaN=True, remove_all_zeroes=False, remove_any_zeroes=False,
                  sort_keys=True):
    """ convert dictionary to numpy array of features
        remove_NaN = True will convert "NaN" string to 0.0
        remove_all_zeroes = True will omit any data points for which
            all the features you seek are 0.0
        remove_any_zeroes = True will omit any data points for which
            any of the features you seek are 0.0
        sort_keys = True sorts keys by alphabetical order. Setting the value as
            a string opens the corresponding pickle file with a preset key
            order (this is used for Python 3 compatibility, and sort_keys
            should be left as False for the course mini-projects).
        NOTE: first feature is assumed to be 'poi' and is not checked for
            removal for zero or missing values.
    """

    return_list = []
    text_message = []

    # Key order - first branch is for Python 3 compatibility on mini-projects,
    # second branch is for compatibility on final project.
    if isinstance(sort_keys, str):
        import pickle
        keys = pickle.load(open(sort_keys, "rb"))
    elif sort_keys:
        keys = sorted(dictionary.keys())
    else:
        keys = dictionary.keys()

    for key in keys:
        tmp_list = []
        text_message.append(dictionary[key]['text'])
        for feature in features:
            try:
                dictionary[key][feature]
            except KeyError:
                print("error: key ", feature, " not present")
                return
            value = dictionary[key][feature]
            if value == "NaN" and remove_NaN:
                value = 0
            tmp_list.append(float(value))

        # Logic for deciding whether or not to add the data point.
        append = True
        # exclude 'poi' class as criteria.
        if features[0] == 'label':
            test_list = tmp_list[1:]
        else:
            test_list = tmp_list

        ### if all features are zero and you want to remove
        ### data points that are all zero, do that here
        if remove_all_zeroes:
            append = False
            for item in test_list:
                if item != 0 and item != "NaN":
                    append = True
                    break
        ### if any features for a given data point are zero
        ### and you want to remove data points with any zeroes,
        ### handle that here
        if remove_any_zeroes:
            if 0 in test_list or "NaN" in test_list:
                append = False
        ### Append the data point if flagged for addition.
        if append:
            return_list.append(np.array(tmp_list))

    return np.array(return_list), keys, text_message
