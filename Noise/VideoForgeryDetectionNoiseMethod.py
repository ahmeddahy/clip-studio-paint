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
import InputOutput
import FeaturesExtraction
import Classify
from xlwt import Workbook

#GUI Class
Path="C:/Users/dell/PycharmProjects/GP_NOISE\DataSet/2_forged.avi"
#InputOutput_obj=InputOutput.Input(Path)
#InputOutput_obj=InputOutput.Read()

InputOutput_obj=InputOutput.Read(Path)
InputOutput_obj.ReadVideo()
Video=InputOutput_obj.GetVideo()

FeaturesExtraction_obj=FeaturesExtraction.FeaturesExtraction(Video)
FeaturesExtraction_obj.WaveletDenoising()
FeaturesExtraction_obj.CorrelationCoefficient()
Features=FeaturesExtraction_obj.GetCorrelationCoefficient()

print(Features)

Classify_obj=Classify.Classify(Features)
Classify_obj.Thershold()
Video,Result=Classify_obj.Localization(Video)
print(Result)
#output class in inputoutput file
for i in range(1, len(Video)):
    print(i)
    cv2.imshow('out', Video[i])
    cv2.waitKey(0)