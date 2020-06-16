#!/usr/bin/python3
# coding=utf-8

import os
import numpy as np
import argparse
import pickle

from scipy import sparse
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from general_utility import print_accuracy, plot_confusion_matrix, print_plot
from BotDetection.pre_processing import buildDataset, featureFormat, targetFeatureSplit
from BotDetection.en.get_feature import distortionVectorizer, PCA
from BotDetection.en.classifier import voting



input_folder = "../../dataset"

featureList = ["label", "similarity", "len", "web", "lenMedia", "lenRT", "lenMediaRT", "mediaEmoji", "puntivirgola"]
EARLY_BIRDS = 'pan19-author-profiling-earlybirds-20190320/'
FINAL_TEST = 'pan19-author-profiling-test-2019-04-29/'

# Build Dataset
# x = buildDataset(input_folder + os.sep, "truth-train.txt", "english")
x = buildDataset(input_folder + os.sep, "truth-train.txt", "english")
# y = buildDataset(input_folder + os.sep + EARLY_BIRDS, "truth.txt", "english")
y = buildDataset(input_folder + os.sep, "truth-dev.txt", "english")

# y = buildDataset(input_folder + os.sep, "truth-dev.txt", "english")

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

# Normalization and scaling
scaler = StandardScaler()  # The idea behind StandardScaler is that it will transform your data such that its distribution will have a mean value 0 and standard deviation of 1.
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)


# Dimensionality Reduction

features_train, features_test, pca = PCA(features_train, features_test)

# Classifier

clf = voting(features_train, labels_train)
predVoting = clf.predict(features_test)

#----------

plot_confusion_matrix(predVoting, labels_test,["human", "bot"])



# Print result


EPOCHS = 25
BATCH_SIZE = 32

print_accuracy(predVoting, labels_test)
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import *
def build_model():
    model = Sequential()
    model.add(Dense(256, input_dim=features_train.shape[1], activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(160, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


model = build_model()
history = model.fit(features_train, np.array(labels_train), epochs=25,
                    validation_data=(features_test, np.array(labels_test)), use_multiprocessing=True,
                    shuffle=True, batch_size=32)
loss, acc = model.evaluate(features_test, np.array(labels_test))

pred = model.predict(features_test).round()

#----------

plot_confusion_matrix(pred=pred, labels=labels_test, target_names=["human", "bot"])
print_plot(history)


