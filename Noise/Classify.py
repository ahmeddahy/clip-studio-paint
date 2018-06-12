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
import Filters
from xlwt import Workbook

class Classify:
      def __init__(self):
          self.CorrelationValues=[]

      def __init__(self,CorrelationValues):
          self.CorrelationValues=CorrelationValues

      def SetCorrelationValues(self,CorrelationValues):
          '''
          input:
          take list of correlation values for all frames
          operation:
          set correlation values
          '''
          self.CorrelationValues=CorrelationValues

      def GaussianMixture(self):
          '''
          operation:
          classify video as forged or original
          return:
          booln value true if forged video or false if original video
          '''
          cnt=0;old_frame_idx=-1;Max=0
          for frame in range(len(self.CorrelationValues)): #classify each frame as forged or not
           Model = GaussianMixtureClassifier.GaussianModel() #construct model of GMD
           Blocks=[]
           for row in range (len(self.CorrelationValues[frame])): #reshape the correlation value 2D array to 1D array
            for col in range (len(self.CorrelationValues[frame][0])):
                Blocks.append(np.array(self.CorrelationValues[frame][row][col])[0])
           Classes = GaussianMixtureClassifier.GaussianMixtureClassifier(Model, Blocks)#classify frame to forged and original regions
           avg1=0;avg2=0   #calcluate the avg of forged class and avg of original class
           if(len(Classes[0])):
            avg1=sum(Classes[0])/len(Classes[0])
           if(len(Classes[1])):
            avg2=sum(Classes[1])/len(Classes[1])
           if (avg1 >= 0.9 or avg2 >= 0.9): #check if avg of forged class >0.9 and calcluate length of consecutive forged frames
             if(old_frame_idx==frame-1):
                cnt = cnt + 1
             else:
                cnt = 1
             old_frame_idx=frame
           Max=max(cnt,Max)
          if(Max>30): # return true if there are 30 consecutive forged frames if the video
           return True
          else:
            return False