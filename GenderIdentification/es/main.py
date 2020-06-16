# !/usr/bin/python3

from os import sep

from GenderIdentification.pre_processing import buildDataset
from GenderIdentification.es.get_feature import TFIDF, LSA
from GenderIdentification.es.classifier import SupportVectorMachine
from general_utility import print_accuracy, plot_confusion_matrix, confusion_matrix

input_folder = "../../dataset" + sep + "es" + sep
input_folder_test = "/home/andrea/PycharmProjects/Bot-and-Gender-Detection-of-Twitter-Accounts/"
FINAL_TEST = "/home/andrea/PycharmProjects/Bot-and-Gender-Detection-of-Twitter-Accounts/dataset/pan19-author-profiling-earlybirds-20190320/es/"
FINAL_TEST = "/home/andrea/PycharmProjects/Bot-and-Gender-Detection-of-Twitter-Accounts/dataset/pan19-author-profiling-test-2019-04-29/es/"

# BUILD DATASET
autori_train, label_train, feature_train, newFeatureTrain, originalText = buildDataset("truth-train.txt",
                                                                                       "spanish", input_folder,
                                                                                       type='Gender')
autori_test, label_test, feature_test, newFeatureTest, originalTextTest = buildDataset("truth-dev.txt",
                                                                                       "spanish",
                                                                                       input_folder, type='Gender')
# FEATURE EXTRACTION
feature_train, feature_test, tfidf = TFIDF(newFeatureTrain, newFeatureTest)


# DIMENSIONALITY REDUCTION
feature_train, feature_test, svd = LSA(feature_train, feature_test)


# CLASSIFICATION
clf = SupportVectorMachine()
clf.fit(feature_train, label_train)
pred = clf.predict(feature_test)
print_accuracy(pred, label_test)


