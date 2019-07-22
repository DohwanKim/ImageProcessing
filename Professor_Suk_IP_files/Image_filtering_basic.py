import numpy as np
from scipy import signal, misc
import matplotlib.pyplot as plt
from scipy import ndimage

def Im_filtering(im, Filter, FilterSize):
    row, col = im.shape
    Padding = int(FilterSize / 2)
    Image_Buffer = np.zeros(shape=(row + 2 * Padding, col + 2 * Padding), dtype=np.uint8)
    Image_Buffer[Padding:row + Padding, Padding:col + Padding] = im[:, :]
    Image_New = np.zeros(shape=(row, col), dtype=np.uint8)
    for y in range(Padding, row + Padding):
        for x in range(Padding, col + Padding):
            buff = Image_Buffer[y - Padding:y + Padding + 1, x - Padding:x + Padding + 1]
            pixel = np.sum(buff * Filter)
            pixel = np.uint8(np.where(pixel > 255, 255, np.where(pixel < 0, 0, pixel)))
            Image_New[y - Padding, x - Padding] = pixel

    return Image_New

lena = misc.imread('lena_256.bmp')
FilterSize = 11
MeanFilter = np.ones(shape=(FilterSize,FilterSize))
MeanFilter = MeanFilter/np.sum(MeanFilter)

Image_New = Im_filtering(lena, MeanFilter, FilterSize)

plt.subplot(121), plt.imshow(lena), plt.gray(), plt.axis('off')
plt.subplot(122), plt.imshow(Image_New), plt.gray(), plt.axis('off')
plt.show()












