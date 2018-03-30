import cv2
import numpy
import numpy as np
import pywt
import math
from Sure_Shrink import Sure_Shrink
import matplotlib.pyplot as plt
def dwt_sure_idwt(img):
    size = img.shape
    '''newsize = max(size[0], size[1])
    if newsize % 2 == 1:
        newsize = newsize + 1
    img = cv2.resize(img, (newsize, newsize))'''
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coeffs = pywt.wavedec2(gray_image, 'sym8', level=4)
    [cA,(cH4,cV4,cD4),(cH3,cV3,cD3),(cH2,cV2,cD2),(cH1,cV1,cD1)]=coeffs
    cD4=Sure_Shrink(cD4)
    cD3=Sure_Shrink(cD3)
    cD2=Sure_Shrink(cD2)
    cD1=Sure_Shrink(cD1)
    res = pywt.waverec2([cA,(cH4,cV4,cD4),(cH3,cV3,cD3),(cH2,cV2,cD2),(cH1,cV1,cD1)], 'sym8')
    g = gray_image - res
    '''plt.imshow(res, cmap='gray')
    plt.show()
    plt.imshow(gray_image, cmap='gray')
    plt.show()
    plt.imshow(g, cmap='gray')
    plt.show()
    plt.imshow(res, cmap='gray')
    plt.show()
    plt.imshow(gray_image, cmap='gray')
    plt.show()
    plt.imshow(g, cmap='gray')
    plt.show()
    cv2.imshow('image2', g)
    cv2.waitKey(0)'''
    return g