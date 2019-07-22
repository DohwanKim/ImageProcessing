import numpy as np
from scipy import signal, misc
import matplotlib.pyplot as plt
from scipy import ndimage

def Gaussian_Filter(sigma, FilterSize):

    G = np.zeros(shape=(FilterSize,FilterSize), dtype = np.float)
    for y in range(-4, 5):
        for x in range(-4, 5):
            s = 1/(2*np.pi*pow(sigma,2))
            v = -(pow(x,2)+pow(y,2))/(2*pow(sigma,2))
            G[y+4, x+4] = s*np.exp(v)
    return(G)

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
FilterSize = 9
sigma = np.arange(1.0, 6.0, 1.0)
plt.subplot(231), plt.imshow(lena), plt.gray(), plt.axis('off')
plt.title('Original Lena Image')
for i in range(len(sigma)):
    G = Gaussian_Filter(sigma[i], FilterSize)
    Image_New = Im_filtering(lena, G, FilterSize)
    plt.subplot(2,3,i+2), plt.imshow(Image_New), plt.gray(), plt.axis('off')
    plt.title('Sigma= %i' %(i+1))
plt.show()