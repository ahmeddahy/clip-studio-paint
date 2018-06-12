import cv2
import numpy
import numpy as np
import pywt
import math
import scipy
import matplotlib.pyplot as plt
import os
import bisect
import DWT_SURE_IDWT
import Sure_Shrink
import GaussianMixtureClassifier
import CalcCrossCorrelation
from xlwt import Workbook

class Filters:

    def NoiseFilter(Frame):
        '''
        :param: 2D array of binary value(0 or 1)
        :operation: remove noise
        :return: 2D array of binary value
        '''
        times = 50
        while (times > 0):
            times = times - 1
            for row in range(1, len(Frame)-1):
                for col in range(1, len(Frame[0])-1):
                    # check if sum of 4 neighbors <2 set the value to 0
                    if Frame[row - 1][col] + Frame[row + 1][col] + Frame[row][col - 1] + Frame[row][col + 1] <2:
                        Frame[row][col] = 0
        return Frame