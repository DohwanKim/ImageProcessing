import numpy as np
from scipy import signal, misc
import matplotlib.pyplot as plt
from scipy import ndimage

bin8 = lambda x : ''.join(reversed( [str((x >> i) & 1) for i in range(8)] ) )

def BitPlane_Slice(value):
    c = np.zeros(8)
    n = 7
    bits = bin8(value)
    for i in range(8):
        p = pow(2, n)
        c[i] = p * int(bits[i])
        n = n - 1
    return c

im = misc.imread('lena_256.bmp')
copyright = misc.imread('copyright.bmp')
row, col = im.shape
Num_BitSlice = 8
Image_BitPlane = np.ndarray(shape=(Num_BitSlice, row, col), dtype=np.uint8)
Image_Restore = np.zeros(shape=(row, col), dtype=np.uint8)

for y in range(row):
    for x in range(col):
        value = im[y, x]
        c = BitPlane_Slice(value)

        for i in range(Num_BitSlice):
            Image_BitPlane[i,y,x] = c[i]

Image_BitPlane[7,:,:] = copyright

for i in range(Num_BitSlice):
    img = Image_BitPlane[i, :, :]
    plt.subplot(2,4,i+1), plt.gray(), plt.axis('off'), plt.imshow(img)
    Image_Restore = img + Image_Restore
plt.show()

plt.subplot(121), plt.gray(), plt.axis('off'), plt.imshow(im), plt.title('Lena')
plt.subplot(122), plt.gray(), plt.axis('off'), plt.imshow(Image_Restore), plt.title('Watermarking Image')
plt.show()

misc.imsave('lena_w.jpg', Image_Restore)






















