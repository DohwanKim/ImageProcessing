import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, signal, misc

def Im_filtering(im, Filter, FilterSize, dummyNum):
    row, col = im.shape
    Padding = int(FilterSize/2)
    Image_Buffer = np.zeros(shape=(row+2*Padding, col+2*Padding), dtype=np.uint8)
    Image_Buffer[Padding:row+Padding, Padding:col+Padding] = im[:, :]
    Image_New = np.zeros(shape=(row, col), dtype=np.uint8)
    for y in range(Padding, row+Padding):
        for x in range(Padding, col+Padding):
            buff = Image_Buffer[y-Padding:y+Padding+1, x-Padding:x+Padding+1]
            pixel = np.sum(buff * Filter) + dummyNum
            pixel = np.uint8(np.where(pixel>255, 255, np.where(pixel<0, 0, pixel)))
            Image_New[y-Padding, x-Padding] = pixel
    return Image_New

lena = misc.imread('./image_sample/medical.bmp')
row, col = lena.shape

laplacian_Filter = np.array([
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]])

Unsharp_Filter = np.array([
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]])

a1 = 0.8
a2 = 1.0
a3 = 1.2

highBoost_Filter_a1 = np.array([
    [-1, -1, -1],
    [-1, 9+a1, -1],
    [-1, -1, -1]])

highBoost_Filter_a2 = np.array([
    [-1, -1, -1],
    [-1, 9+a2, -1],
    [-1, -1, -1]])

highBoost_Filter_a3 = np.array([
    [-1, -1, -1],
    [-1, 9+a3, -1],
    [-1, -1, -1]])

laplacian_Filter_size = 3

Image_laplacian = Im_filtering(lena, laplacian_Filter, laplacian_Filter_size, 0)
Image_Unsharp = Im_filtering(lena, Unsharp_Filter, laplacian_Filter_size, 0)
Image_highBoost01 = Im_filtering(lena, highBoost_Filter_a1, laplacian_Filter_size, 0)
Image_highBoost02 = Im_filtering(lena, highBoost_Filter_a2, laplacian_Filter_size, 0)
Image_highBoost03 = Im_filtering(lena, highBoost_Filter_a3, laplacian_Filter_size, 0)

plt.subplot(3,2,1)
plt.gray()
plt.axis('off')
plt.imshow(lena)
plt.title('Original Images')

plt.subplot(3,2,2)
plt.gray()
plt.axis('off')
plt.imshow(Image_laplacian)
plt.title('laplacian Filter Images')

plt.subplot(3,2,3)
plt.gray()
plt.axis('off')
plt.imshow(Image_Unsharp)
plt.title('Unsharp Filter Images')

plt.subplot(3,2,4)
plt.gray()
plt.axis('off')
plt.imshow(Image_highBoost01)
plt.title('High-Boost Filter Images (alpha :'+str(a1)+')')

plt.subplot(3,2,5)
plt.gray()
plt.axis('off')
plt.imshow(Image_highBoost02)
plt.title('High-Boost Filter Images (alpha :'+str(a2)+')')

plt.subplot(3,2,6)
plt.gray()
plt.axis('off')
plt.imshow(Image_highBoost03)
plt.title('High-Boost Filter Images (alpha :'+str(a3)+')')

plt.show()
