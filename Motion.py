import cv2
import numpy as np
from typing import List
from sklearn import svm


class Motion:
    def __init__(self, svm_classifier: svm.SVC):
        self.__frames: List[np.ndarray] = []
        self.__base_frame: np.ndarray = None
        self.__svm_classifier = svm_classifier
        self.__features = []
        self.__fps = 0.0

    def read_video(self, path: str):
        count = 0
        cap = cv2.VideoCapture(path)
        self.__fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                # Our operations on the frame come here
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = np.array(gray).astype(int)
                self.__frames.append(gray)
                count += 1
                if count == 1:
                    self.__base_frame = gray
                else:
                    self.__base_frame += gray
            else:
                break
        self.__base_frame = (self.__base_frame/count).round().astype(int)
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def __clip(minimum, maximum, value):
        if value > maximum:
            return maximum
        elif value < minimum:
            return minimum
        else:
            return value

    def __markov_features(self, frame: np.ndarray)-> list:
        m_list = []
        R = frame - self.__base_frame
        h, w = R.shape
        Rh = np.zeros((h - 1, w))
        Rv = np.zeros((h, w - 1))
        Rd = np.zeros((h - 1, w - 1))
        Rm = np.zeros((h - 1, w - 1))
        for u in range(0, h):
            for v in range(0, w):
                if u != h - 1:
                    Rh[u, v] = self.__clip(-4, 4, R[u, v] - R[u + 1, v])
                if v != w - 1:
                    Rv[u, v] = self.__clip(-4, 4, R[u, v] - R[u, v + 1])
                if u != h - 1 and v != w - 1:
                    Rd[u, v] = self.__clip(-4, 4, R[u, v] - R[u + 1, v + 1])
                    Rm[u, v] = self.__clip(-4, 4, R[u + 1, v] - R[u, v + 1])
        Mh = np.zeros((9, 9))
        Mv = np.zeros((9, 9))
        Md = np.zeros((9, 9))
        Mm = np.zeros((9, 9))

        ###################################################

        for i in range(0, 9):
            for j in range(0, 9):
                sum_numerator = [0.0, 0.0, 0.0, 0.0]
                sum_denominator = [0.0, 0.0, 0.0, 0.0]
                for u in range(0, h):
                    for v in range(0, w):
                        if u < h - 2:
                            if Rh[u, v] == i - 4 and Rh[u + 1, v] == j - 4:
                                sum_numerator[0] += 1
                        if u < h - 1:
                            if Rh[u, v] == i - 4:
                                sum_denominator[0] += 1
                        if v < w - 2:
                            if Rv[u, v] == i - 4 and Rv[u, v + 1] == j - 4:
                                sum_numerator[1] += 1
                        if v < w - 1:
                            if Rv[u, v] == i - 4:
                                sum_denominator[1] += 1
                        if u < h - 2 and v < w - 2:
                            if Rd[u, v] == i - 4 and Rd[u + 1, v + 1] == j - 4:
                                sum_numerator[2] += 1
                            if Rm[u + 1, v] == i - 4 and Rm[u, v + 1] == j - 4:
                                sum_numerator[3] += 1
                        if u < h - 1 and v < w - 1:
                            if Rd[u, v] == i - 4:
                                sum_denominator[2] += 1
                            if Rm[u, v] == i - 4:
                                sum_denominator[3] += 1
                if sum_denominator[0] == 0.0:
                    Mh[i, j] = 0.0
                else:
                    Mh[i, j] = sum_numerator[0] / sum_denominator[0]
                if sum_denominator[1] == 0.0:
                    Mv[i, j] = 0.0
                else:
                    Mv[i, j] = sum_numerator[1] / sum_denominator[1]
                if sum_denominator[2] == 0.0:
                    Md[i, j] = 0.0
                else:
                    Md[i, j] = sum_numerator[2] / sum_denominator[2]
                if sum_denominator[3] == 0.0:
                    Mm[i, j] = 0.0
                else:
                    Mm[i, j] = sum_numerator[3] / sum_denominator[3]
                m_list.append((Mh[i, j] + Mv[i, j] + Md[i, j] + Mm[i, j])/4.0)
        return m_list

    def compute_features(self):
        for frame in self.__frames:
            self.__features.append(self.__markov_features(frame))

    def get_fake_time(self)-> list:
        seconds = []
        classification = self.__svm_classifier.predict(self.__features)
        n = classification.__len__()
        count = 0
        for i in range(0, n):
            if classification[i] == 0:
                count += 1
            else:
                count = 0
            if count >= self.__fps:
                seconds.append((i + 1)/self.__fps)
                count = 0
        return seconds
