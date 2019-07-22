import numpy as np
from scipy import signal, misc
import matplotlib.pyplot as plt
from scipy import ndimage

#Get Image
aero = misc.imread('./Day03_images/aero2.bmp')  #일반 사진
hole = misc.imread('./Day03_images/hole.bmp')   #구멍 모양의 이미지
lena = misc.imread('./image_sample/DongRyn.bmp')   #레나 파일

#각 파일을 실수단위로 변환 (정확한 연산을 위해서)
aero = np.float32(aero)
hole = np.float32(hole)
lena = np.float32(lena)

#두 값을 더하고 255이상의 오버플로우 값은 255으로 해준다.
im_new1 = np.uint8(np.where(lena+aero>255, 255, lena+aero))
im_new2 = np.uint8(np.where(lena+hole>255, 255, lena+hole))

plt.subplot(231)
plt.imshow(np.uint(aero))
plt.gray()
plt.title('AERO image')

plt.subplot(232)
plt.imshow(np.uint(lena))
plt.gray()
plt.title('LENA image')

plt.subplot(233)
plt.imshow(im_new1)
plt.gray()
plt.title('AERO + LENA')

plt.subplot(234)
plt.imshow(np.uint(lena))
plt.gray()
plt.title('AERO image')

plt.subplot(235)
plt.imshow(np.uint(hole))
plt.gray()
plt.title('Hole image')

plt.subplot(236)
plt.imshow(im_new2)
plt.gray()
plt.title('AERO + Hole')

plt.show()

### 3-2 랜덤한 가우시안 노이즈가 낀 같은 이미지8개를 불러온다.

#Get Image
noise1 = misc.imread('./Day03_images/noise1.bmp')
noise2 = misc.imread('./Day03_images/noise2.bmp')
noise3 = misc.imread('./Day03_images/noise3.bmp')
noise4 = misc.imread('./Day03_images/noise4.bmp')
noise5 = misc.imread('./Day03_images/noise5.bmp')
noise6 = misc.imread('./Day03_images/noise6.bmp')
noise7 = misc.imread('./Day03_images/noise7.bmp')
noise8 = misc.imread('./Day03_images/noise8.bmp')

#각 실수형으로 변환
noise1 = np.float32(noise1)
noise2 = np.float32(noise2)
noise3 = np.float32(noise3)
noise4 = np.float32(noise4)
noise5 = np.float32(noise5)
noise6 = np.float32(noise6)
noise7 = np.float32(noise7)
noise8 = np.float32(noise8)

#전부 다 더해서 평균을 내준다.
noise_average = (noise1 + noise2 + noise3 + noise4 + noise5 + noise6 + noise7 + noise8)/8

plt.subplot(331)
plt.imshow(np.uint(noise1))
plt.gray()
plt.title('noise1')

plt.subplot(332)
plt.imshow(np.uint(noise2))
plt.gray()
plt.title('noise2')

plt.subplot(333)
plt.imshow(np.uint(noise3))
plt.gray()
plt.title('noise3')

plt.subplot(334)
plt.imshow(np.uint(noise4))
plt.gray()
plt.title('noise4')

plt.subplot(335)
plt.imshow(np.uint(noise5))
plt.gray()
plt.title('noise5')

plt.subplot(336)
plt.imshow(np.uint(noise6))
plt.gray()
plt.title('noise6')

plt.subplot(337)
plt.imshow(np.uint(noise7))
plt.gray()
plt.title('noise7')

plt.subplot(338)
plt.imshow(np.uint(noise8))
plt.gray()
plt.title('noise1')


plt.subplot(339)
plt.imshow(noise_average)
plt.gray()
plt.title('Noise_average')

plt.show()


### 3-2 (used For)

im_new3 = np.zeros(shape=(256, 256))
NumOfImage = 8
for i in range(NumOfImage):
    filename = 'noise' + str(i+1) + '.bmp'
    im = misc.imread('./Day03_images/'+filename)
    plt.subplot(2,4,i+1)
    plt.imshow(im)
    plt.gray()
    plt.title('Noisy Image: %i' %(i+1))
    plt.axis('off')
plt.show()
plt.imshow(np.uint8(im_new3))
plt.title('Mean Image of Noisy Images')

### 3-3, 3-4
#Get Image
hole2 = misc.imread('./Day03_images/hole2.bmp')
lena2 = misc.imread('./image_sample/DongRyn.bmp')
diff1 = misc.imread('./Day03_images/diff1.bmp')
diff2 = misc.imread('./Day03_images/diff2.bmp')

hole2 = np.float32(hole2)
lena2 = np.float32(lena2)
diff1 = np.float32(diff1)
diff2 = np.float32(diff2)

im_new01 = np.where(hole2-lena2<0, 0, hole2-lena2)
im_new02 = np.where(lena2-hole2<0, 0, lena2-hole2)
im_new03 = np.abs(np.where(diff1-diff2<0, 0, diff1-diff2))


plt.subplot(131)
plt.imshow(np.uint(im_new01))
plt.gray()
plt.title('hole - lena')

plt.subplot(132)
plt.imshow(np.uint(im_new02))
plt.gray()
plt.title('lena - hole')

plt.subplot(133)
plt.imshow(np.uint(im_new03))
plt.gray()
plt.title('diff1-diff2')

plt.show()

### 3-5
lena02 = misc.imread('./image_sample/lena_256.bmp')
gray127 = misc.imread('./Day03_images/gray127.bmp')
gray128 = misc.imread('./Day03_images/gray128.bmp')

#logical_AND = lena & gray128
#logical_OR = lena & gray

logical_AND = np.bitwise_and(lena02, gray128)
logical_OR = np.bitwise_or(lena02, gray127)