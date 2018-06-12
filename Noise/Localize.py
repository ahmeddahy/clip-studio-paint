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

class Localize:
      def __init__(self):
          self.CorrelationValues=[]

      def __init__(self,CorrelationValues):
          self.CorrelationValues=CorrelationValues
      def SetCorrelationValues(self):
          self.CorrelationValues=[]

      def Thershold(self):
          '''
          operation: for each frame thershold the correlation value to 1 if value>=0.9 otherwise 0
          '''
          self.BinarizedCorrelationValues=[]
          for Blocks in range(0, len(self.CorrelationValues)):
              BinarizedCorrelationValues = [[0 for i in range(len(self.CorrelationValues[Blocks][0])+2)] for j in range(len(self.CorrelationValues[Blocks])+2)]
              for row in range(0, len(self.CorrelationValues[Blocks])): # for each correlation values frame
                  for col in range(0, len(self.CorrelationValues[Blocks][row])): # for each correlation value in the frame
                      binaryvalue = 1
                      if (self.CorrelationValues[Blocks][row][col] < .9): #check if value <0.9 to thershold it to 0
                        binaryvalue = 0
                      BinarizedCorrelationValues[row+ 1][col + 1] = binaryvalue # store the thershold value
              self.BinarizedCorrelationValues.append(BinarizedCorrelationValues) #store all the binarized frame of blocks in list
          self.PreProcessing()

      def PreProcessing(self):
          '''
          operation:
          remove noises from correlation values frame in all video frames
          '''
          self.SmoothedCorrelationValues=[]
          for Blocks in range(0, len(self.BinarizedCorrelationValues)):
              SmoothedCorrelationValues=Filters.Filters.NoiseFilter(self.BinarizedCorrelationValues[Blocks]) #remove noise from frame
              self.SmoothedCorrelationValues.append(SmoothedCorrelationValues) #list of lists to store the smoothed correlation values


      def ConnectedComponent(self,Frame):
          '''
          :param Frame: 2D array of correlation value
          :operation: find connected components
          :return: connected components
          '''
          blocks = np.uint8(Frame)
          ret, thresh = cv2.threshold(blocks, 0, 255, 0)
          im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
          return  contours


      def DrawBox(self,ForgedFrames,Video):
          '''
          :param ForgedFrames: list of lists of frame number and coordinates of forged region
          :param Video: video frames
          :operation: draw rectangle around the forged region if the video is forged
          :return: video and result
          '''
          Result="Original"
          if len(ForgedFrames) > 0:
              Result="Forged"
              for i in range(0, len(ForgedFrames)):
                  avgx = 0
                  avgy = 0
                  avgw = 0
                  avgh = 0
                  k = ForgedFrames[i]
                  for j in range(0, len(k)):
                      avgx += k[j][1][0]
                      avgy += k[j][1][1]
                      avgw += k[j][1][2]
                      avgh += k[j][1][3]
                  avgx /= (len(k))
                  avgy /= (len(k))
                  avgw /= (len(k))
                  avgh /= (len(k))
                  avgx = int(avgx)
                  avgy = int(avgy)
                  avgw = int(avgw)
                  avgh = int(avgh)
                  for j in range(0, len(k)):
                      cv2.rectangle(Video[k[j][0]], (avgx, avgy), (avgx + avgw, avgy + avgh), (0, 255, 0), 2)
          return  Video,Result


      def Localization(self,Video):
          '''
          :param Video: actual video frames
          :operation: localize the forged region in the video
          :return: video and result
          '''
          tmp=[]
          List_forged = []
          for Blocks in range(0, len(self.SmoothedCorrelationValues)):
              contours=self.ConnectedComponent(self.SmoothedCorrelationValues[Blocks])#find all objects in the frame
              idx = 0
              MaxArea = 0
              for contour in range(0, len(contours), 1):
               area = cv2.contourArea(contours[contour])
               if (MaxArea < area): #take the largest object in the frame
                  MaxArea = area
                  idx = contour
              if (MaxArea < 100):# ignore very small objects
                 if len(tmp) > 10:
                   List_forged.append(tmp)
                 tmp = []
              else:
                  x, y, w, h = cv2.boundingRect(contours[idx])#find coordinates
                  coor = [(x * 8), (y * 8), w * 8, h * 8]  #calcluate the coordinates in actual frame
                  tmp.append([Blocks + 1, coor])
          if(len(tmp)>10):
              List_forged.append(tmp)
          return self.DrawBox(List_forged,Video) # draw rectangel
