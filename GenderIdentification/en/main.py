# !/usr/bin/python3

from os import sep
from sklearn.metrics import confusion_matrix
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import *

from GenderIdentification.pre_processing import buildDataset
from GenderIdentification.en.get_feature import TFIDF, truncated_SVD
from GenderIdentification.en.classifier import SupportVectorMachine
from general_utility import print_accuracy, plot_confusion_matrix

input_folder = "/home/andrea/PycharmProjects/Bot-and-Gender-Detection-of-Twitter-Accounts/dataset" + sep + "en" + sep
input_folder_test = "/home/andrea/PycharmProjects/Bot-and-Gender-Detection-of-Twitter-Accounts/"
FINAL_TEST = "/home/andrea/PycharmProjects/Bot-and-Gender-Detection-of-Twitter-Accounts/dataset/pan19-author-profiling-test-2019-04-29/en/"
EARLY_BIRD = "/home/andrea/PycharmProjects/Bot-and-Gender-Detection-of-Twitter-Accounts/dataset/pan19-author-profiling-earlybirds-20190320/en/"

# BUILD DATASET
autori_train, label_train, feature_train, newFeatureTrain, originalText = buildDataset("truth-train.txt",
                                                                                       "english", input_folder,
                                                                                       type='Gender')
autori_test, label_test, feature_test, newFeatureTest, originalTextTest = buildDataset("truth-dev.txt",
                                                                                       "english",
                                                                                       input_folder, type='Gender')
# FEATURE EXTRACTION
feature_train, feature_test, tfidf = TFIDF(feature_train, feature_test)
print(feature_test.shape)
print(feature_train.shape)
# DIMENSIONALITY REDUCTION
svd, feature_train = truncated_SVD(feature_train)
feature_test = svd.transform(feature_test)
print(feature_test.shape)
print(feature_train.shape)
# CLASSIFICATION
clf = SupportVectorMachine()
clf.fit(feature_train, label_train)
pred = clf.predict(feature_test)
print_accuracy(pred, label_test)

cm = confusion_matrix(label_test, pred)

# ----------

plot_confusion_matrix(cm, ["man", "woman"])


# Print result
'''
def build_model():
    model = Sequential()
    model.add(Dense(256, input_dim=feature_train.shape[1], activation='relu'))
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
history = model.fit(feature_train, np.array(label_train), epochs=15, use_multiprocessing=True,
                    shuffle=True, batch_size=1)
loss, acc = model.evaluate(feature_test, np.array(label_test))
print("acc net: ", acc)
print("loss net: ", loss)
pred_net = model.predict(feature_test).round()

cm = confusion_matrix(label_test, pred_net)

# ----------

plot_confusion_matrix(cm,["man", "woman"])
print_accuracy(pred_net, label_test, target=["man", "woman"])
'''

