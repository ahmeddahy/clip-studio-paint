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
        self.__base_frame = (self.__base_frame / count).round().astype(int)
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
    def counti (arr,h,w):
        count=np.zeros(9)
        for i in range (0,h):
            for j in range (0,w):
                count[arr[i][j]+4]+=1
        return count
    def countdimension (arr,rowc1,colc1,rowc2,colc2,h,w):
        count=np.zeros((9,9))
        for i in range (0,h):
            for j in range (0,w):
                count[arr[i+rowc1][j+colc1]+4][arr[i+rowc2][j+colc2]+4]+=1
        return count
    def __markov_features(self, frame: np.ndarray) -> list:
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
        Rhcounti = self.counti(Rh,h-1,w)
        Rvcounti = self.counti(Rv,h,w-1)
        Rdcounti = self.counti(Rd, h-1, w - 1)
        Rmcounti = self.counti(Rm, h-1, w - 1)
        Rhcountdimension =self.countdimension(Rh,0,0,1,0,h-2,w)
        Rvcountdimension = self.countdimension(Rv, 0, 0, 0, 1, h, w-2)
        Rdcountdimension = self.countdimension(Rd, 0, 0, 1, 1, h - 2, w-2)
        Rmcountdimension = self.countdimension(Rm, 1, 0, 0, 1, h - 2, w-2)
        for i in range(-4,5):
            for j in range (-4,5):
                numeratorh=Rhcountdimension[i+4][j+4]
                denominatorh=Rhcounti[i+4]
                if denominatorh == 0.0:
                    Mh[i + 4][j + 4] = 0.0
                else:
                    Mh[i + 4][j + 4] = numeratorh / denominatorh
                ##############################################
                numeratorv = Rvcountdimension[i + 4][j + 4]
                denominatorv = Rvcounti[i + 4]
                if denominatorv == 0.0:
                    Mv[i + 4][j + 4] = 0.0
                else:
                    Mv[i + 4][j + 4] = numeratorv / denominatorv
                ##############################################
                numeratord = Rdcountdimension[i + 4][j + 4]
                denominatord = Rdcounti[i + 4]
                if denominatord == 0.0:
                    Md[i + 4][j + 4] = 0.0
                else:
                    Md[i + 4][j + 4] = numeratord / denominatord
                ##############################################
                numeratorm = Rmcountdimension[i + 4][j + 4]
                denominatorm = Rmcounti[i + 4]
                if denominatorm == 0.0:
                    Mm[i + 4][j + 4] = 0.0
                else:
                    Mm[i + 4][j + 4] = numeratorm / denominatorm
                m_list.append((Mh[i, j] + Mv[i, j] + Md[i, j] + Mm[i, j]) / 4.0)
        return m_list
    def compute_features(self):
        for frame in self.__frames:
            self.__features.append(self.__markov_features(frame))

    def get_fake_time(self) -> list:
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
                seconds.append((i + 1) / self.__fps)
                count = 0
        return seconds
