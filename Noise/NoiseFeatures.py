import numpy as np
import pywt
import math
import cv2
import bisect


class NoiseFeatures:
    def __init__(self):
        self.pure_noise = []

    def dwt_sure_idwt(self, img):  # input : frame
        size = img.shape  # get the size of this frame
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert image to gray (Preprocessind)
        coeffs = pywt.wavedec2(gray_image, 'sym8', level=4)  # apply wavelet decomposition for 4 levels
        [cA, (cH4, cV4, cD4), (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1,
                                                                 cD1)] = coeffs  # Get the approximate Coefficients(LL) and detail coefficients for each level(LH,HL,HH)
        # Apply SURE Shrink to HH coefficients for each level
        cD4 = self.Sure_Shrink(cD4)
        cD3 = self.Sure_Shrink(cD3)
        cD2 = self.Sure_Shrink(cD2)
        cD1 = self.Sure_Shrink(cD1)
        res = pywt.waverec2([cA, (cH4, cV4, cD4), (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)],
                            'sym8')  # reconstruct image without noise
        self.pure_noise = gray_image - res  # get the noise features of this frame


    def Sure_Shrink(self, coefficients):
        count = 0  # variable for try lambda
        minsure = 1e18  # variable to minimize the SURE equation
        t = 0  # the lambda that minimized SURE equation
        size = coefficients.shape  # get the size of coefficients
        # convert coefficients from 2D to 1D and put the absolute value in allcoeff
        numofcoefficients = size[0] * size[1]
        allcoeff = []
        com_sum = 0
        com_sum_coeff = []
        for listcoeff in coefficients:
            for coeff in listcoeff:
                allcoeff.append(abs(coeff))
        allcoeff.sort()  # sort this list
        for coeff in allcoeff:  # get the commulative sum of value^2 in allcoeff and put it in com_sum_coeff
            com_sum = com_sum + (abs(coeff) * abs(coeff))
            com_sum_coeff.append(com_sum)
        limit = math.sqrt(2 * math.log10(numofcoefficients))  # the limit for trying lambda
        while (count <= limit):
            sum = 0
            ind = bisect.bisect_right(allcoeff, count)  # get the index of abs(coeff)that is greater than count
            num_gt = len(com_sum_coeff) - ind  # number of abs(coeff)s that are greater than count
            sum = sum + (num_gt * count * count)  # apply the euation of SURE Shrink
            if (ind > 0):
                sum = sum + com_sum_coeff[ind - 1]
            sure = numofcoefficients + sum - 2 * ind
            if sure < minsure:  # Minimization condition for SURE equation
                minsure = sure
                t = count
            count = count + .01  # add small value to try more lambdas
        coefficients = pywt.threshold(coefficients, t, 'soft')  # apply soft thrshold using lambda(t)

        return coefficients
