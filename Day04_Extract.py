import numpy as np
from scipy import signal, misc
import matplotlib.pyplot as plt
from scipy import ndimage
#LSB에 워터마크가 들어간 이미지를 다시 불러들여 분해.

def BitPlane_Slice(value):
    bin8 = lambda x: ''.join(reversed([str((x >> i) & 1) for i in range(8)]))
    c = np.zeros(8)
    n = 7
    bits = bin8(value)
    for i in range(8):
        p = pow(2, n)
        c[i] = p * int(bits[i])
        n = n - 1
    return c

im = misc.imread('./image_result/DongRyn_w.png')
row, col = im.shape

Num_BitSlice = 8

Image_BitPlane = np.ndarray(shape=(Num_BitSlice, row, col), dtype=np.uint8) #비트평면 분할 배열 256x256 (각 인덱스 8의 배열)
Image_Restore = np.zeros(shape=(row, col), dtype=np.uint8)

for y in range(row):
    for x in range(col):
        value = im[y, x]
        c = BitPlane_Slice(value)

        for i in range(Num_BitSlice):
            Image_BitPlane[i, y, x] = c[i]

Image_Restore = Image_BitPlane[7, :, :]

plt.subplot(1,2,1)
plt.gray()
plt.imshow(im)
plt.title('Original watermark Image')

plt.subplot(1,2,2)
plt.gray()
plt.imshow(Image_Restore)
plt.title('Restore watermark Image')
plt.show()
