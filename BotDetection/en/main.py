#!/usr/bin/python3
# coding=utf-8

import os
import numpy as np

from scipy import sparse
from sklearn.preprocessing import StandardScaler

from general_utility import print_accuracy, plot_confusion_matrix, print_plot
from BotDetection.pre_processing import buildDataset, featureFormat, targetFeatureSplit
from BotDetection.en.get_feature import distortionVectorizer, PCA
from BotDetection.en.classifier import bot_neural_net, train_neural_net

input_folder = "../../dataset"

featureList = ["label", "similarity", "len", "web", "lenMedia", "lenRT", "lenMediaRT", "mediaEmoji", "puntivirgola", "compound", "negative", "hasHashtag"]

TRAIN_DEV = 'pan19-author-profiling-training-2019-02-18/'
EARLY_BIRDS = 'pan19-author-profiling-earlybirds-20190320/'
FINAL_TEST = 'pan19-author-profiling-test-2019-04-29/'


# Build Dataset
x = buildDataset(input_folder + os.sep + TRAIN_DEV, "truth-train.txt", "english")
y = buildDataset(input_folder + os.sep + TRAIN_DEV, "truth-dev.txt", "english")


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
labels_train = np.array(labels_train)
labels_test = np.array(labels_test)
features_train = sparse.hstack([features_train, text_TrainDistortion]).todense()
features_test = sparse.hstack([features_test, text_TestDistortion]).todense()

# Normalization and scaling
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

# Dimensionality Reduction
features_train, features_test, pca = PCA(features_train, features_test)

# Classifier
input_shape = features_train.shape[1]
model = bot_neural_net(input_shape=input_shape)  # instantiate the model

history = train_neural_net(model, features_train, labels_train, features_test, labels_test)  # fit the model

loss, acc = model.evaluate(features_test, labels_test)  # test the model

pred = model.predict(features_test).round()

# ----------

plot_confusion_matrix(pred=pred, labels=labels_test, target_names=["human", "bot"])
print_plot(history)
