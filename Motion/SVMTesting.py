from sklearn import svm
import numpy


def classifyFrames(fake_train, orig_train):
    origLabel= [1] * len(orig_train)
    sizeLabel=len(fake_train)+len(origLabel)
    trainingLabels=numpy.zeros((sizeLabel, 1))
    trainingLabels[0:len(origLabel), 0] = origLabel

    trainingData=numpy.zeros((len(orig_train)+len(fake_train), len(orig_train[0])))
    trainingData[0:len(orig_train), :] = orig_train
    trainingData[len(orig_train): len(orig_train)+len(fake_train), :] = fake_train

    # correct = 0
    clf = svm.SVC(kernel='poly', max_iter=1000, gamma=3.4, degree=8)
    clf.fit(trainingData, trainingLabels)
    # classification_labels = clf.predict(testing_data)
    #
    # for i in range(len(classification_labels)):
    #     if testing_label[i] == 'original' and classification_labels[i] == 1:
    #         correct += 1
    #     if testing_label[i] == 'fake' and classification_labels[i] == 0:
    #         correct += 1
    return clf

    # print(100*correct/len(classification_labels))
