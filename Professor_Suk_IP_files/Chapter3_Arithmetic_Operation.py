import numpy as np
from scipy import signal, misc
import matplotlib.pyplot as plt
from scipy import ndimage

#Image Read
aero = misc.imread('aero2.bmp')
hole = misc.imread('hole.bmp')
lena = misc.imread('lena_256.bmp')
aero = np.float32(aero)
hole = np.float32(hole)
lena = np.float32(lena)

im_new1 = np.uint8(np.where(lena+aero>255, 255, lena+aero))
im_new2 = np.uint8(np.where(lena+hole>255, 255, lena+hole))

plt.subplot(231), plt.imshow(np.uint8(aero)), plt.gray(), plt.title('AERO Image'), plt.axis('off')
plt.subplot(232), plt.imshow(np.uint8(lena)), plt.gray(), plt.title('LENA Image'), plt.axis('off')
plt.subplot(233), plt.imshow(im_new1), plt.gray(), plt.title('ADD Result'), plt.axis('off')
plt.subplot(234), plt.imshow(np.uint8(lena)), plt.gray(), plt.title('LENA Image'), plt.axis('off')
plt.subplot(235), plt.imshow(np.uint8(hole)), plt.gray(), plt.title('HOLE Image'), plt.axis('off')
plt.subplot(236), plt.imshow(im_new2), plt.gray(), plt.title('ADD Result'), plt.axis('off')
plt.show()

#Image Read
lena = misc.imread('lena_256.bmp')
gray128 = misc.imread('gray128.bmp')
gray127 = misc.imread('gray127.bmp')

#logical_AND = lena & gray128
#logical_OR  = lena | gray127

logical_AND = np.bitwise_and(lena, gray128)
logical_OR  = np.bitwise_or(lena, gray127)

im_new3 = np.zeros(shape=(256,256))
NumOfImage = 8
for i in range(NumOfImage):
    filename = 'noise'+str(i+1)+'.bmp'
    im = misc.imread(filename)
    plt.subplot(2,4,i+1), plt.imshow(im),plt.gray()
    plt.title('Noisy Image: %i' %(i+1)), plt.axis('off')
    im_new3 = im_new3 + np.float32(im)/NumOfImage
plt.show()
plt.imshow(logical_AND), plt.title('Mean Image of Noisy Images'), plt.axis('off')
plt.show()













