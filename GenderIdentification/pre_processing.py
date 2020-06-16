#!/usr/bin/python3


import xml.etree.ElementTree as ET
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import es_core_news_sm
import spacy

def buildDataset(filename, language, path, type='Bot'):
    nlp = None
    if language == "spanish":
        print("faccio lemma")
        #nlp = es_core_news_sm.load()
        spacy.prefer_gpu()
        nlp = spacy.load("en_core_web_sm")
        nlp.disable_pipes(["tagger", "parser", "ner"])
    else:
        from nltk.stem import WordNetLemmatizer
        wordnet_lemmatizer = WordNetLemmatizer()
    print(language)
    stemmer = SnowballStemmer(language)
    authorList = []
    labelTrain = []
    featureTrain = []
    featureLemma = []
    featurePOS = []
    i = -1
    originalText = []
    newFeature = {}
    with open(path + filename, "r") as file:
        for line in file:
            truth_column = line.split(":::")
            author = truth_column[0]
            isBot = truth_column[1]
            if isBot == 'bot':
                continue
            elif isBot == "human":
                i += 1
                authorList.append(author)
                gender = truth_column[2][0]
                featureTrain.append("")
                featureLemma.append("")
                originalText.append("")
                featurePOS.append("")
                if gender == "m":
                    if type == 'Gender':
                        labelTrain.append(0)
                    else:
                        labelTrain.append(1)
                elif gender == "f":
                    if type == 'Gender':
                        labelTrain.append(1)
                    else:
                        labelTrain.append(2)

            root = ET.parse(path + "/" + author + '.xml').getroot()
            dizionario = processXML(path + "/" + author + '.xml')
            for lineXML in root.iter('document'):
                tweet = lineXML.text

                y = tweet.split()
                if language == 'english':
                    for word in y:
                        # sostituisci:  sito e numeri
                        featureTrain[i] += (stemmer.stem(word) + " ")  # 0.8274193548
                        featureLemma[i] += wordnet_lemmatizer.lemmatize(word) + " "
                        originalText[i] += word

                elif language == 'spanish':
                    for token in nlp(tweet):
                        featureLemma[i] += (token.lemma_ + " ")  # 0.8274193548
                        featureTrain[i] += stemmer.stem(token.text) + " "
                        originalText[i] += token.text

                newFeature.update({author: dizionario})
    print(len(labelTrain))
    return authorList, labelTrain, featureTrain, featureLemma, originalText


def processXML(filename):
    cntRetweet = 0
    cnt = 0
    cntWebsite = 0
    cntLun = 0
    cntLun2 = 0
    numWebsite = 0
    numEmoji = 0
    virgole = 0
    punti = 0
    puntivirgola = 0
    root = ET.parse(filename).getroot()
    cntEnter = 0
    for lineXML in root.iter('document'):
        x = lineXML.text
        numWebsite += x.count("https://")
        numWebsite += x.count("http://")
        virgole += x.count(", ")
        puntivirgola += x.count("; ")
        punti += x.count(". ")
        cnt += 1
        cntLun += len(x)
        cntEnter = x.count("\n")
        y = lineXML.text.split()
        if "RT @" in x:
            cntRetweet += 1
            cntLun2 += len(x)
        if "https://" in x or "http://" in x or "www." in x or ".com" in x:
            cntWebsite += 1
    dict = {"punti": punti, "puntivirgola": puntivirgola, "virgola": virgole, "RT": cntRetweet, "len": cntLun,
            "web": cntWebsite, "lenMedia": cntLun / cnt if cnt > 0 else 0,
            "lenRT": cntLun2, "lenMediaRT": cntLun2 / cntRetweet if cntRetweet > 0 else 0, "numEmoji": numEmoji,
            "mediaEmoji": numEmoji / cnt if cnt > 0 else 0, "cnt": cnt,
            "percentWeb": ((1.0 * cntWebsite) / cnt) * 100 if cnt > 0 else 0, "cntEnter": cntEnter}
    return dict
