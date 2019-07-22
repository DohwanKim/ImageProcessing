import numpy as np
from scipy import signal, misc
import matplotlib.pyplot as plt
from scipy import ndimage

#Image Read
im = misc.imread('lena_256.bmp')
row, col = im.shape

#중간강조
n=50
N1 = 100
N2 = 130
im_new = np.ndarray(shape=(row,col), dtype=np.uint8)
for y in range(row):
    for x in range(col):
        value = im[y,x]
        if value > N1 and value < 130:
            im_new[y,x] = value + n
        else:
            im_new[y,x] = value

im_new1 = np.where(im < 100, im, np.where(im > 130,im, im+n))



plt.subplot(121), plt.imshow(im), plt.title('Original Image'), plt.gray(), plt.axis('off')
plt.subplot(122), plt.imshow(im_new), plt.title('Image'), plt.gray(), plt.axis('off')
plt.show()




