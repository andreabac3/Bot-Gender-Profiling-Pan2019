#!/usr/bin/python3


from sklearn.svm import SVC


def SupportVectorMachine():
    clf = SVC(kernel='rbf', probability=True, C=5000, random_state=0)
    return clf



