import numpy as np
from scipy import signal, misc
import matplotlib.pyplot as plt

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

lena = misc.imread('./image_sample/lena_256.bmp')
row, col = lena.shape
value = 0

#영상의 각 픽셀의 8비트의 흑백 값을 불러옴
for y in range(row):
    for x in range(col):
        # 데이터를 value 전부 더해줌
        value = value + lena[y, x]

#픽셀 갯수만큼 나눠준다.
lena_average = int(value/(row*col))
#(a)
print(lena_average)

#이진화 된 영상을 담아줄 변수
#(b)
im_binary = np.ndarray(shape=(row, col), dtype=np.uint8)
for y in range(row):
    for x in range(col):
        value = lena[y, x]
        #평균값보다 크면 1, 아니면 0
        if value > lena_average:
            im_binary[y, x] = 1
        else:
            im_binary[y, x] = 0

plt.subplot(211)
plt.imshow(lena)
plt.axis('off')
plt.gray()
plt.title('Lena Default')

plt.subplot(212)
plt.axis('off')
plt.imshow(im_binary)
plt.gray()
plt.title('Lena_binary')
plt.show()

#이하로 (c)
Num_BitSlice = 8
Image_BitPlane = np.ndarray(shape=(Num_BitSlice, row, col), dtype=np.uint8)

for y in range(row):
    for x in range(col):
        value = lena[y, x]
        c = BitPlane_Slice(value)   #비트평면 분할 함수 적용

        #비트평면 분할 된 것을 3차원 배열에 담아준다.
        #픽셀당 8개의 인덱스이니 for문은 8번 돌아간다.
        for i in range(Num_BitSlice):
            Image_BitPlane[i, y, x] = c[i]

Image_BitPlane[7, :, :] = im_binary

Image_Restore = np.zeros(shape=(row, col), dtype=np.uint8)

for i in range(Num_BitSlice):
    #0~7자리 마다 각 성분
    img = Image_BitPlane[i, :, :]
    print(img)
    Image_Restore = img + Image_Restore
    plt.subplot(2, 4, i+1)
    plt.imshow(img)
    plt.gray()
    plt.title('BitSlice Image: %i' % (i + 1))
    plt.axis('off')
plt.show()

plt.subplot(1,2,1)
plt.axis('off')
plt.imshow(lena)
plt.title('Original Images')

plt.subplot(1,2,2)
plt.axis('off')
plt.imshow(Image_Restore)
plt.title('Restore Images')
plt.show()

#워터마크를 삽입하고 복원한 이미지 파일
misc.imsave('./image_result/lena_w.png', Image_Restore)

#이하로 (d)
im = misc.imread('./image_result/lena_w.png')
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
plt.axis('off')
plt.gray()
plt.imshow(im)
plt.title('Original watermarking Image')

plt.subplot(1,2,2)
plt.axis('off')
plt.gray()
plt.imshow(Image_Restore)
plt.title('Restore watermark Image in Original')
plt.show()