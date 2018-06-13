import cv2
import numpy
import numpy as np
import pywt
import math
import scipy
import matplotlib.pyplot as plt
import os
import bisect
import GaussianMixtureClassifier
from xlwt import Workbook


class Input:
    def __init__(self):
        self.VideoPath = ""

    def __init__(self, Path):
        self.VideoPath = Path

    def SetPath(self, Path):
        self.VideoPath = Path

    def GetPath(self):
        return self.VideoPath


class Output:
    def __init__(self):
        self.Video = ""
        self.Result = ""

    def __init__(self, Video, Result):
        self.Video = Video
        self.Result = Result

    def GetVideo(self):
        return self.Video

    def GetResult(self):
        return self.Result


class Read:
    def __init__(self):
        self.Path = Input.GetPath()

    def __init__(self, Path):
        self.Path = Path

    def ReadVideo(self):
        self.Video = cv2.VideoCapture(self.Path)
        self.VideoFrames = []
        while (self.Video.isOpened()):
            ret, frame = self.Video.read()
            if ret:
                self.VideoFrames.append(frame)
            else:
                break
        self.Video.release()
        cv2.destroyAllWindows()

    def GetVideo(self):
        return self.VideoFrames
