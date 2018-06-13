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
import NoiseFeatures
import CrossCorrelation
from xlwt import Workbook


class FeaturesExtraction:
    def __init__(self):
        self.VideoFrames = []

    def __init__(self, VideoFrames):
        self.VideoFrames = VideoFrames

    def WaveletDenoising(self):
        '''
        operation:
        denoising all video frames using wavelet shrinkage
        '''
        DWT_SURE_IDWT = NoiseFeatures.NoiseFeatures()
        self.DenoisedFrames = []
        for frame in range(len(self.VideoFrames)):
            DWT_SURE_IDWT.dwt_sure_idwt(self.VideoFrames[frame])  # denoise the frame using wavelet shrinkage
            DenoisedFrame = DWT_SURE_IDWT.pure_noise
            self.DenoisedFrames.append(DenoisedFrame)  # list of list contain of all denoised frames

    def CorrelationCoefficient(self):
        '''
        operation:
        calculate correlation values between each consecutive frames
        '''
        self.CorrelationValues = []
        for frame in range(len(self.DenoisedFrames) - 1):
            # calculate cross-correlation for each 2 consecutive frame
            CrossCorrelation_obj = CrossCorrelation.CrossCorrelation()
            CorrelationValues = CrossCorrelation_obj.calc_corr(self.DenoisedFrames[frame],
                                                               self.DenoisedFrames[frame + 1])
            self.CorrelationValues.append(CorrelationValues)  # list of list contain of all correlation values

    def GetCorrelationCoefficient(self):
        '''
        operation:
        send correlation values
        return:
        correlation values
        '''
        return self.CorrelationValues
