import cv2
import numpy as np
from typing import List
from sklearn import svm
from numba import jit

from ReadFile import Reader
from SVMTesting import SVMClassifier


class Motion:
    def __init__(self):
        self.__frames: List[np.ndarray] = []
        self.__base_frame: np.ndarray = None
        self.__svm_classifier: svm.SVC = None
        self.__features = []
        self.__fps = 0.0
        ###########################
        self.__Rh = None
        self.__Rv = None
        self.__Rd = None
        self.__Rm = None
        self.__Mh = np.zeros((9, 9))
        self.__Mv = np.zeros((9, 9))
        self.__Md = np.zeros((9, 9))
        self.__Mm = np.zeros((9, 9))
        self.__Rhcounti = None
        self.__Rvcounti = None
        self.__Rdcounti = None
        self.__Rmcounti = None
        self.__Rhcountdimension = None
        self.__Rvcountdimension = None
        self.__Rdcountdimension = None
        self.__Rmcountdimension = None

    @staticmethod
    def __clip(minimum, maximum, value):
        if value > maximum:
            return maximum
        elif value < minimum:
            return minimum
        else:
            return value

    def countValue(self, arr, h, w):
        count = np.zeros(9)
        for i in range(0, h):
            for j in range(0, w):
                count[np.int(
                    arr[i][j] + 4)] += 1  # for each direction residue frame count the values between -4 and 4 inclusive
        return count

    def calcTransitionProbability(self, arr, rowc1, colc1, rowc2, colc2, h, w):
        count = np.zeros((9, 9))
        for i in range(0, h):
            for j in range(0,
                           w):  # calculate the probability of transition between 2 values around 9 states for each direction
                count[np.int(arr[i + rowc1][j + colc1] + 4)][np.int(arr[i + rowc2][j + colc2] + 4)] += 1
        return count

    def __calcFeatures(self, h, w):
        self.__Rhcounti = self.countValue(self.__Rh, h - 1, w)
        self.__Rvcounti = self.countValue(self.__Rv, h, w - 1)
        self.__Rdcounti = self.countValue(self.__Rd, h - 1, w - 1)
        self.__Rmcounti = self.countValue(self.__Rm, h - 1, w - 1)
        self.__Rhcountdimension = self.calcTransitionProbability(self.__Rh, 0, 0, 1, 0, h - 2, w)
        self.__Rvcountdimension = self.calcTransitionProbability(self.__Rv, 0, 0, 0, 1, h, w - 2)
        self.__Rdcountdimension = self.calcTransitionProbability(self.__Rd, 0, 0, 1, 1, h - 2, w - 2)
        self.__Rmcountdimension = self.calcTransitionProbability(self.__Rm, 1, 0, 0, 1, h - 2, w - 2)

    def __computeDirectiones(self, frameResidue):
        h, w = frameResidue.shape
        # initialize 4 2D-array for 4 directions
        self.__Rh = np.zeros((h - 1, w))  # horizontal direction
        self.__Rv = np.zeros((h, w - 1))  # vertical direction
        self.__Rd = np.zeros((h - 1, w - 1))  # diagonal direction
        self.__Rm = np.zeros((h - 1, w - 1))  # minor-diagonal direction
        for u in range(0, h):
            for v in range(0, w):
                if u != h - 1:
                    self.__Rh[u, v] = self.__clip(-4, 4, frameResidue[u, v] - frameResidue[
                        u + 1, v])  # limiting the values between -4 and 4
                if v != w - 1:
                    self.__Rv[u, v] = self.__clip(-4, 4, frameResidue[u, v] - frameResidue[u, v + 1])
                if u != h - 1 and v != w - 1:
                    self.__Rd[u, v] = self.__clip(-4, 4, frameResidue[u, v] - frameResidue[u + 1, v + 1])
                    self.__Rm[u, v] = self.__clip(-4, 4, frameResidue[u + 1, v] - frameResidue[u, v + 1])

    def markov_features(self, frame: np.ndarray) -> list:
        features = []
        frameResidue = frame - self.__base_frame  # calculate frame residue
        h, w = frameResidue.shape
        self.__computeDirectiones(frameResidue)  # construct 4 directions
        self.__calcFeatures(h, w)  # calculate transition probability
        for i in range(-4, 5):  # loop from -4 to 4 inclusive to include all 9 states
            for j in range(-4, 5):
                numeratorh = self.__Rhcountdimension[i + 4][j + 4]
                denominatorh = self.__Rhcounti[i + 4]
                if denominatorh == 0.0:
                    self.__Mh[i + 4][j + 4] = 0.0
                else:
                    self.__Mh[i + 4][j + 4] = numeratorh / denominatorh
                numeratorv = self.__Rvcountdimension[i + 4][j + 4]
                denominatorv = self.__Rvcounti[i + 4]
                if denominatorv == 0.0:
                    self.__Mv[i + 4][j + 4] = 0.0
                else:
                    self.__Mv[i + 4][j + 4] = numeratorv / denominatorv
                numeratord = self.__Rdcountdimension[i + 4][j + 4]
                denominatord = self.__Rdcounti[i + 4]
                if denominatord == 0.0:
                    self.__Md[i + 4][j + 4] = 0.0
                else:
                    self.__Md[i + 4][j + 4] = numeratord / denominatord
                numeratorm = self.__Rmcountdimension[i + 4][j + 4]
                denominatorm = self.__Rmcounti[i + 4]
                if denominatorm == 0.0:
                    self.__Mm[i + 4][j + 4] = 0.0
                else:
                    self.__Mm[i + 4][j + 4] = numeratorm / denominatorm
                features.append((self.__Mh[i + 4, j + 4] + self.__Mv[i + 4, j + 4] + self.__Md[i + 4, j + 4] +
                                 self.__Mm[i + 4, j + 4]) / 4.0)
        return features

    def calcBaseFrame(self, videoFrames: list) -> np.ndarray:
        framesSize = len(videoFrames)
        base_frame: np.ndarray
        count = 0
        for frame in videoFrames:  # get average pixels of all frames to get base frame
            if (count == 0):
                base_frame = frame
                count = 1
            else:
                base_frame = base_frame + frame

        base_frame = (base_frame / framesSize).round().astype(int)
        return base_frame

    def computeFrameFeatures(self, path):
        self.__frames, self.__fps = Reader.read_video(path)  # get video frames and frame rate
        self.__base_frame = self.calcBaseFrame(self.__frames)  # get base frame
        i = 0
        for frame in self.__frames:
            self.__features.append(self.markov_features(frame))  # calculate Markove features for each frame
            print(i)
            i = i + 1

    def getFakeTime(self) -> list:
        fake_train, orig_train = Reader.readTrainingFeatures()  # get fake and original training features
        classifier = SVMClassifier()
        classifier.trainData(fake_train, orig_train)  # trained SVM classifier
        self.__svm_classifier = classifier.clf
        classification = self.__svm_classifier.predict(self.__features)  # classify input video frames
        seconds = []  # list holds fake starting point in video
        n = classification.__len__()
        count = 0

        for i in range(0, n):
            if classification[i] == 0:  # check if classified frame is forged
                count += 1
            elif count > 0:
                count -= 1
            if count >= self.__fps:  # add forged second
                seconds.append((i + 1) / self.__fps)
                count = 0
        return seconds

