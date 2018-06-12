from sklearn import svm
import numpy

class SVMClassifier:
    def __init__(self):
        self.clf:svm.SVC  = None

    def trainData(self, fake_train, orig_train):
        origLabel= [1] * len(orig_train)                        # list of 1's with length of original train
        sizeLabel=len(fake_train)+len(origLabel)
        trainingLabels=numpy.zeros((sizeLabel, 1))
        trainingLabels[0:len(origLabel), 0] = origLabel         # construct training labels 1 for original and 0 for fake
        trainingData=numpy.zeros((len(orig_train)+len(fake_train), len(orig_train[0])))
        trainingData[0:len(orig_train), :] = orig_train

        # construct a list where original training data comes first then fake training data comes next
        trainingData[len(orig_train): len(orig_train)+len(fake_train), :] = fake_train
        self.clf = svm.SVC(kernel='poly', max_iter=476, gamma=0.6, degree=4)            # set SVM parameters
        self.clf.fit(trainingData, trainingLabels)                                      # train the SVM


