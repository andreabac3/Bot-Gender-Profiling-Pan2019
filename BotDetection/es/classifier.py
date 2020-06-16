#!/usr/bin/python3


from sklearn.svm import SVC


def SVM(features_train, label_train):
    clf = SVC(kernel='rbf', probability=True, random_state=0, gamma='auto')
    clf.fit(features_train, label_train)
    return clf

