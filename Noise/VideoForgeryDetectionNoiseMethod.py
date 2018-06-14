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
import InputOutput
import FeaturesExtraction
import ForgeryDetermination
import Localize
from xlwt import Workbook

# GUI Class
Path = "F:/4th year/Graduation project/Video-Forgery-Detection/Data set/12_forged.avi"
# InputOutput_obj=InputOutput.Input(Path)
# InputOutput_obj=InputOutput.Read()

InputOutput_obj = InputOutput.Read(Path)
InputOutput_obj.ReadVideo()
Video = InputOutput_obj.GetVideo()

FeaturesExtraction_obj = FeaturesExtraction.FeaturesExtraction(Video)
FeaturesExtraction_obj.WaveletDenoising()

FeaturesExtraction_obj.CorrelationCoefficient()
Features = FeaturesExtraction_obj.GetCorrelationCoefficient()

Classify_obj = ForgeryDetermination.Classify(Features)
forged = Classify_obj.GaussianMixture()

if (forged == True):
    Localize_obj = Localize.Localize(Features)
    Localize_obj.Thershold()
    Video, Result = Localize_obj.Localization(Video)
else:
    Result = "Original"
# output class in inputoutput file
for i in range(1, len(Video)):
    print(i)
    cv2.imshow('out', Video[i])
    cv2.waitKey(0)
print(Result)
