#영상처리 강의자료 04
# <실습 5-3>

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, misc
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D


#예제 01
sigma = 1.0
FilterSize = 9
G = np.zeros(shape=(FilterSize, FilterSize), dtype=np.float)
for x in range(-4, 5):
    for y in range(-4, 5):
        s = 1/(2*np.pi*pow(sigma, 2))
        v = -(pow(x,2)+pow(y,2))/(2*pow(sigma,2))
        G[y+4, x+4] = s*np.exp(v)
        print('{:.4f}'.format(G[y+4, x+4]), end=' ')
    print('~~~~')

plt.imshow(G)
plt.gray()
plt.show()

#예제 03 - (d)
#필터 함수 (원본 데이터, 필터, 필터 사이즈, 결과에 취해 줄 정수)
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

#9x9 가우시안 필터
def Gaussian_Filter_9x9(FilterSize, sigma):
    G_new = np.zeros(shape=(FilterSize, FilterSize), dtype=np.float)
    for x in range(-4, 5):
        for y in range(-4, 5):
            s = 1/(2*np.pi*pow(sigma, 2))
            v = -(pow(x,2)+pow(y,2))/(2*pow(sigma, 2))
            G_new[y+4, x+4] = s*np.exp(v)
    return G_new

#(필터사이즈, 시그마) 파라미터를 가우시안 필터
def Gaussian_Filter(FilterSize, sigma):
    G_new = np.zeros(shape=(FilterSize, FilterSize), dtype=np.float)
    F_halfsize = int(FilterSize/2)
    print(F_halfsize)
    for x in range(-F_halfsize, F_halfsize+1):
        for y in range(-F_halfsize, F_halfsize+1):
            s = 1/(2*np.pi*pow(sigma, 2))
            v = -(pow(x,2)+pow(y,2))/(2*pow(sigma, 2))
            G_new[y+F_halfsize, x+F_halfsize] = s*np.exp(v)
            print('{:.4f}'.format(G[y + 4, x + 4]), end=' ')
    return G_new

#sample source
lena = misc.imread('./image_sample/DongRyn.bmp')
row, col = lena.shape

FilterSize = 9
sumdummy = 0

G1 = np.zeros(shape=(FilterSize, FilterSize), dtype=np.float)
G2 = np.zeros(shape=(FilterSize, FilterSize), dtype=np.float)
G3 = np.zeros(shape=(FilterSize, FilterSize), dtype=np.float)
G4 = np.zeros(shape=(FilterSize, FilterSize), dtype=np.float)
G5 = np.zeros(shape=(FilterSize, FilterSize), dtype=np.float)

G1 = Gaussian_Filter_9x9(FilterSize, 1.0)
G2 = Gaussian_Filter_9x9(FilterSize, 2.0)
G3 = Gaussian_Filter_9x9(FilterSize, 3.0)
G4 = Gaussian_Filter_9x9(FilterSize, 4.0)
G5 = Gaussian_Filter(FilterSize, 5.0)

G1_result = Im_filtering(lena, G1, FilterSize, sumdummy)
G2_result = Im_filtering(lena, G2, FilterSize, sumdummy)
G3_result = Im_filtering(lena, G3, FilterSize, sumdummy)
G4_result = Im_filtering(lena, G4, FilterSize, sumdummy)
G5_result = Im_filtering(lena, G5, FilterSize, sumdummy)


plt.subplot(2,3,1)
plt.gray()
plt.axis('off')
plt.imshow(lena)
plt.title('Original Images')

plt.subplot(2,3,2)
plt.gray()
plt.axis('off')
plt.title('Gaussian Images #sigma: 1.0')
plt.imshow(G1_result)

plt.subplot(2,3,3)
plt.gray()
plt.axis('off')
plt.title('Gaussian Images #sigma: 2.0')
plt.imshow(G2_result)

plt.subplot(2,3,4)
plt.gray()
plt.axis('off')
plt.title('Gaussian Images #sigma: 3.0')
plt.imshow(G3_result)

plt.subplot(2,3,5)
plt.gray()
plt.axis('off')
plt.title('Gaussian Images #sigma: 4.0')
plt.imshow(G4_result)

plt.subplot(2,3,6)
plt.gray()
plt.axis('off')
plt.title('Gaussian Images #sigma: 5.0')
plt.imshow(G5_result)

plt.show()
