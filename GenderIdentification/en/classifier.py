#!/usr/bin/python3


from sklearn.svm import SVC


def SupportVectorMachine():
    clf = SVC(kernel='rbf', probability=True, C=12000)
    return clf

