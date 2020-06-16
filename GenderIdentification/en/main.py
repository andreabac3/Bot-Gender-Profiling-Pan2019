# !/usr/bin/python3

from os import sep
from sklearn.metrics import confusion_matrix

from GenderIdentification.pre_processing import buildDataset
from GenderIdentification.en.get_feature import TFIDF, truncated_SVD
from GenderIdentification.en.classifier import SupportVectorMachine
from general_utility import print_accuracy, plot_confusion_matrix

TRAIN_DEV = "../../dataset/pan19-author-profiling-training-2019-02-18" + sep + "en" + sep
FINAL_TEST = "../../dataset/pan19-author-profiling-test-2019-04-29/en/"
EARLY_BIRD = "../../dataset/pan19-author-profiling-earlybirds-20190320/en/"

# BUILD DATASET
autori_train, label_train, feature_train, newFeatureTrain, originalText = buildDataset("truth-train.txt",
                                                                                       "english", TRAIN_DEV,
                                                                                       type='Gender')
autori_test, label_test, feature_test, newFeatureTest, originalTextTest = buildDataset("truth-dev.txt",
                                                                                       "english",
                                                                                       TRAIN_DEV, type='Gender')
# FEATURE EXTRACTION
feature_train, feature_test, tfidf = TFIDF(feature_train, feature_test)



# DIMENSIONALITY REDUCTION
svd, feature_train = truncated_SVD(feature_train)
feature_test = svd.transform(feature_test)



# CLASSIFICATION
clf = SupportVectorMachine()
clf.fit(feature_train, label_train)
pred = clf.predict(feature_test)
print_accuracy(pred, label_test)

cm = confusion_matrix(label_test, pred)

# ----------

plot_confusion_matrix(cm, ["man", "woman"])
