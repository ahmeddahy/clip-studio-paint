import cv2
import numpy
import numpy as np
import pywt
import math
import bisect
def Sure_Shrink(coefficients):
    count = 0
    minsure=1e18
    t=0
    size=coefficients.shape
    numofcoefficients=size[0]*size[1]
    allcoeff=[]
    com_sum=0
    com_sum_coeff=[]
    for listcoeff in coefficients:
        for coeff in listcoeff:
            allcoeff.append(abs(coeff))
    allcoeff.sort()
    for coeff in allcoeff:
        com_sum = com_sum + (abs(coeff)*abs(coeff))
        com_sum_coeff.append(com_sum)
    limit=math.sqrt(2 * math.log10(numofcoefficients))
    while(count<=limit):
        sum=0
        ind=bisect.bisect_right(allcoeff,count)
        num_gt=len(com_sum_coeff)-ind
        sum=sum+(num_gt*count*count)
        if(ind>0):
            sum=sum+com_sum_coeff[ind-1]
        sure = numofcoefficients + sum - 2 * ind
        if sure < minsure:
            minsure = sure
            t = count
        count=count+.01
    coefficients = pywt.threshold(coefficients, t, 'hard')
    #coefficients = pywt.threshold(coefficients, count, 'soft')

    return coefficients