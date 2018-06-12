from numba import jit
from numpy import arange

from Motion import *

m=Motion()
m.computeFrameFeatures('09_forged.avi')
seconds=m.get_fake_time3()
print(seconds)
