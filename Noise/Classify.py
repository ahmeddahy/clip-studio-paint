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
      def SetCorrelationValues(self):
          self.CorrelationValues=[]
      def Thershold(self):
          self.BinarizedCorrelationValues=[]
          for Blocks in range(0, len(self.CorrelationValues)):
              BinarizedCorrelationValues = [[0 for i in range(len(self.CorrelationValues[Blocks][0])+2)] for j in range(len(self.CorrelationValues[Blocks])+2)]
              for row in range(0, len(self.CorrelationValues[Blocks])):
                  for col in range(0, len(self.CorrelationValues[Blocks][row])):
                      binaryvalue = 1
                      if (self.CorrelationValues[Blocks][row][col] < .9):
                        binaryvalue = 0
                      BinarizedCorrelationValues[row+ 1][col + 1] = binaryvalue
              self.BinarizedCorrelationValues.append(BinarizedCorrelationValues)
          self.PreProcessing()

      def PreProcessing(self):
          self.SmoothedCorrelationValues=[]
          workbook = Workbook()
          sheet1 = workbook.add_sheet("Sheet 1")
          for Blocks in range(0, len(self.BinarizedCorrelationValues)):
              f = "sheet" + str(Blocks + 1)
              sheet1 = workbook.add_sheet(f)
              SmoothedCorrelationValues=Filters.Filters.MedFilter(self.BinarizedCorrelationValues[Blocks])
              self.SmoothedCorrelationValues.append(SmoothedCorrelationValues)
              for k in range(0, len(SmoothedCorrelationValues)-2):
                  for m in range(0, len(SmoothedCorrelationValues[k])-2):
                      sheet1.write(k, m, SmoothedCorrelationValues[k + 1][m + 1])
          workbook.save("correlation values.xls")

      def ConnectedComponent(self,Frame):
          blocks = np.uint8(Frame)
          ret, thresh = cv2.threshold(blocks, 0, 255, 0)
          im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
          return  contours
      def DrawBox(self,ForgedFrames,Video):
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
          tmp=[]
          List_forged = []
          for Blocks in range(0, len(self.SmoothedCorrelationValues)):
              contours=self.ConnectedComponent(self.SmoothedCorrelationValues[Blocks])
              idx = 0
              MaxArea = 0
              for contour in range(0, len(contours), 1):
               area = cv2.contourArea(contours[contour])
               if (MaxArea < area):
                  MaxArea = area
                  idx = contour
              if (MaxArea < 100):
                 if len(tmp) > 10:
                   List_forged.append(tmp)
                 tmp = []
              else:
                  x, y, w, h = cv2.boundingRect(contours[idx])
                  coor = [(x * 8), (y * 8), w * 8, h * 8]
                  tmp.append([Blocks + 1, coor])
          if(len(tmp)>10):
              List_forged.append(tmp)
          return self.DrawBox(List_forged,Video)