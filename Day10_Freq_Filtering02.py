import numpy as np
from scipy import signal, misc
import matplotlib.pyplot as plt
from scipy import ndimage

# Image Read
lena = misc.imread('./image_sample/lena_256.bmp')
col, row = lena.shape

# Apply Fourier TransForm
# 이미지 푸리에 변환 -> 주파수 영역 이미지 -> 복구
F = np.fft.fft2(lena)   #fft2 : 2차원 이미지용
Mag = np.abs(F)     #폭(크기성분) -> 주로 이 부분을 변경
Mag = np.fft.fftshift(Mag)
Pha = np.angle(F)   #위상 -> 위상은 거의 건들지 않음


#가우시안 필터
col = 256
row = 256
cx = 128
cy = 128
D0 = 10

Gaussian_L = np.zeros(shape=[col, row], dtype=np.float32)
Gaussian_H = np.zeros(shape=[col, row], dtype=np.float32)

for x in range(col):
    for y in range(row):
        D = np.sqrt((cx - x) ** 2 + (cy - y) ** 2)
        Gaussian_L[x, y] = np.exp(-D**2 / (2*D0**2))

Gaussian_H = 1 - Gaussian_L

#만들어진 필터와 원본 이미지의 Mag와 곱해줌 (필터링)
Mag_Low = Mag*Gaussian_L
Mag_High = Mag*Gaussian_H

#다시 쉬프트
Mag_Low = np.fft.fftshift(Mag_Low)
Mag_High = np.fft.fftshift(Mag_High)

#새로 만들어진 Mag로 복구
Complex_Low = Mag_Low*np.exp(1j*Pha)
Complex_High = Mag_High*np.exp(1j*Pha)

IM_High = np.fft.ifft2(Complex_High)
IM_Low = np.fft.ifft2(Complex_Low)
IM_High = np.real(IM_High)
IM_Low = np.real(IM_Low)


#출력
plt.subplot(221),plt.imshow(Gaussian_L), plt.gray(), plt.axis('off'), plt.title('Gaussian Lowpass')
plt.subplot(222),plt.imshow(Gaussian_H), plt.gray(), plt.axis('off'),plt.title('Gaussian Highpass')
plt.subplot(223),plt.imshow(IM_Low), plt.gray(), plt.axis('off'), plt.title('Gaussian Lowpass Image')
plt.subplot(224),plt.imshow(IM_High), plt.gray(), plt.axis('off'), plt.title('Gaussian Highpass Image')

plt.show()