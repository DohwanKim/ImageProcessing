import numpy as np
from scipy import signal, misc
import matplotlib.pyplot as plt
from scipy import ndimage

#Image Read
lena = misc.imread('lena_256.bmp')
gray128 = misc.imread('gray128.bmp')
gray127 = misc.imread('gray127.bmp')

#logical_AND = lena & gray128
#logical_OR  = lena | gray127

logical_AND = np.bitwise_and(lena, gray128)
logical_OR  = np.bitwise_or(lena, gray127)


