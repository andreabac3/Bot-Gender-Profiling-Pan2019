#!/usr/bin/python3

import os

import pickle
import numpy as np
from scipy import sparse
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from general_utility import print_accuracy, plot_confusion_matrix
from BotDetection.pre_processing import buildDataset, featureFormat, targetFeatureSplit
from BotDetection.es.get_feature import distortionVectorizer, PCA
from BotDetection.es.classifier import SVM

featureList = ["label", "similarity", "len", "web", "lenMedia", "lenRT", "lenMediaRT", "mediaEmoji", "puntivirgola",
               "cntTag3", "cntTag", "cntTAG2", "compound", "negative", "hasHashtag"]
EARLY_BIRDS = 'pan19-author-profiling-earlybirds-20190320/'
FINAL_TEST = 'pan19-author-profiling-test-2019-04-29/'

# Build Dataset
STORED = False
input_folder = "../../dataset"
if not STORED:
    # Build Dataset
    x = buildDataset(input_folder + os.sep, "truth-train.txt", "spanish")
    #x = buildDataset(input_folder + os.sep, "truth.txt", "spanish")
    #y = buildDataset(input_folder + os.sep + FINAL_TEST, "truth.txt", "english")
    y = buildDataset(input_folder + os.sep, "truth-dev.txt", "spanish")
    with open("es_bot_dataset_x.pickle", 'wb') as f:
        pickle.dump(x, f)
    with open("es_bot_dataset_y.pickle.pickle", 'wb') as f:
        pickle.dump(y, f)
else:
    with open("es_bot_dataset_x.pickle", 'rb') as f:
        x = pickle.load(f)
    with open("es_bot_dataset_y.pickle", 'rb') as f:
        y = pickle.load(f)
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


scaler = StandardScaler()  # The idea behind StandardScaler is that it will transform your data such that its distribution will have a mean value 0 and standard deviation of 1.
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

# Dimensionality Reduction
features_train, features_test, pca = PCA(features_train, features_test)

# Classifier

clf = SVM(features_train, labels_train)
predVoting = clf.predict(features_test)

print_accuracy(predVoting, labels_test)

from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import *

features_train2, validation_batches, y_train, y_val = train_test_split(features_train, labels_train, test_size=0.3,
                                                                       random_state=0)

'''
def build_model():
    model = Sequential()
    model.add(Dense(200, input_dim=features_train.shape[1], activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(180, activation='relu'))
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
history = model.fit(features_train, np.array(labels_train), epochs=40,
                    validation_data=(features_test, np.array(labels_test)), use_multiprocessing=True,
                    shuffle=True, batch_size=10)
loss, acc = model.evaluate(features_test, np.array(labels_test))

pred_net = model.predict(features_test).round()
cm = confusion_matrix(labels_test, pred_net)

# ----------

plot_confusion_matrix(cm, ["human", "bot"])
'''