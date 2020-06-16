#!/usr/bin/python3

import os

import pickle
import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from general_utility import print_accuracy
from BotDetection.pre_processing import buildDataset, featureFormat, targetFeatureSplit
from BotDetection.es.get_feature import distortionVectorizer, PCA
from BotDetection.es.classifier import SVM

featureList = ["label", "similarity", "len", "web", "lenMedia", "lenRT", "lenMediaRT", "mediaEmoji", "puntivirgola", "compound", "negative", "hasHashtag"]

TRAIN_DEV = 'pan19-author-profiling-training-2019-02-18/'

EARLY_BIRDS = 'pan19-author-profiling-earlybirds-20190320/'
FINAL_TEST = 'pan19-author-profiling-test-2019-04-29/'

# Build Dataset
input_folder = "../../dataset"

x = buildDataset(input_folder + os.sep + TRAIN_DEV, "truth-train.txt", "spanish")
y = buildDataset(input_folder + os.sep + TRAIN_DEV, "truth-dev.txt", "spanish")

# FEATURE EXTRACTION
dataTrain, KEYS_train, text_Train = featureFormat(x, featureList)
labels_train, features_train = targetFeatureSplit(dataTrain)

dataTest, KEYS_test, text_Test = featureFormat(y, featureList)
labels_test, features_test = targetFeatureSplit(dataTest)

##  Distortion Text
text_TrainDistortion, text_TestDistortion, vectDistortion = distortionVectorizer(text_Train, text_Test)

# Union of Feature
features_train = np.array(features_train)
features_test = np.array(features_test)
features_train = sparse.hstack([features_train, text_TrainDistortion]).todense()
features_test = sparse.hstack([features_test, text_TestDistortion]).todense()

scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

# Dimensionality Reduction
features_train, features_test, pca = PCA(features_train, features_test)

# Classifier

clf = SVM(features_train, labels_train)
predVoting = clf.predict(features_test)

print_accuracy(predVoting, labels_test)
