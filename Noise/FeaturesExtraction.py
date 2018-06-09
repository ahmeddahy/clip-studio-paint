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
class FeaturesExtraction:
      def __init__(self):
          self.VideoFrames=[]
      def __init__(self,VideoFrames):
          self.VideoFrames=VideoFrames
      def WaveletDenoising(self):
          self.DenoisedFrame=[]
          for frame in range(len(self.VideoFrames)):
              DenoisedFrame=DWT_SURE_IDWT.dwt_sure_idwt(self.VideoFrames[frame])
              self.DenoisedFrame.append(DenoisedFrame)
      def CorrelationCoefficient(self):
          self.CorrelationValues=[]
          for frame in range (len(self.DenoisedFrame)-1):
              CorrelationValues=CalcCrossCorrelation.calc_corr(self.DenoisedFrame[frame],self.DenoisedFrame[frame + 1])
              self.CorrelationValues.append(CorrelationValues)
      def GetCorrelationCoefficient(self):
          return self.CorrelationValues