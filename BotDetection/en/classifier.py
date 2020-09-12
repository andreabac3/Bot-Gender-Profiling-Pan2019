#!/usr/bin/python3


from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import *


def adaboost(clf):
    from sklearn.ensemble import AdaBoostClassifier
    return AdaBoostClassifier(clf, n_estimators=30, algorithm='SAMME.R', random_state=0, learning_rate=65)


def voting_ensamble_model(features_train, labels_train):
    from sklearn.ensemble import VotingClassifier
    from sklearn.ensemble import AdaBoostClassifier
    SVM = SVC(kernel='rbf', probability=True, random_state=0)
    votingCLF = VotingClassifier(estimators=[('SV', SVC(kernel='rbf', probability=True, random_state=0, gamma='auto')), ('ADA', AdaBoostClassifier(SVM, n_estimators=30, algorithm='SAMME.R', random_state=0, learning_rate=65))],
                                 voting='soft',
                                 weights=[1, 1], n_jobs=-1)
    votingCLF = CalibratedClassifierCV(votingCLF)
    votingCLF.fit(features_train, labels_train)
    return votingCLF


def SVM():
    return SVC(kernel='rbf', probability=True, random_state=0, gamma='auto')





def bot_neural_net(input_shape):
    model = Sequential()
    model.add(Dense(256, input_dim=input_shape, activation='relu'))
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


def train_neural_net(model, feature, label, valid_feature, valid_label, epochs: int = 25, batch_size: int = 32):
    history = model.fit(feature, label, epochs=epochs, validation_data=(valid_feature, valid_label), use_multiprocessing=True,
                        shuffle=True, batch_size=batch_size)
    return history