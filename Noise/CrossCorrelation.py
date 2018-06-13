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


class CrossCorrelation:
    def calc_corr(self, noise_matrix1, noise_matrix2):  # input : 2 consecuitive frames
        noise_matrix1 = np.float32(noise_matrix1)
        noise_matrix2 = np.float32(noise_matrix2)
        size = noise_matrix1.shape  # get the size of the frame
        blocks = []
        row_size = size[0]
        col_size = size[1]
        sub_matrix1 = numpy.zeros((8, 8))  # bolck in frame 1
        mask_matrix = numpy.zeros((8, 8))  # bolck in frame 2
        # divide 2 frames into 8*8 blocks and apply normalized cross correlation
        for i in range(0, row_size, 8):
            block = []
            for j in range(0, col_size, 8):
                sub_matrix1 = noise_matrix1[i:i + 8, j:j + 8]
                mask_matrix = noise_matrix2[i:i + 8, j:j + 8]
                corr_val = cv2.matchTemplate(mask_matrix, sub_matrix1,
                                             cv2.TM_CCORR_NORMED)  # calculate corr_value between 2 blocks
                block.append(corr_val)
            blocks.append(block)
        return blocks

    def NoiseFilter(Frame):
        '''
        :param: 2D array of binary value(0 or 1)
        :operation: remove noise
        :return: 2D array of binary value
        '''
        times = 50
        while (times > 0):
            times = times - 1
            for row in range(1, len(Frame) - 1):
                for col in range(1, len(Frame[0]) - 1):
                    # check if sum of 4 neighbors <2 set the value to 0
                    if Frame[row - 1][col] + Frame[row + 1][col] + Frame[row][col - 1] + Frame[row][col + 1] < 2:
                        Frame[row][col] = 0
        return Frame
