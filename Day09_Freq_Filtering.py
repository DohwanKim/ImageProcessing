import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, misc

# Image Read
lena = misc.imread('./image_sample/lena_256.bmp')
col, row = lena.shape

# Apply Fourier TransForm
# 이미지 푸리에 변환 -> 주파수 영역 이미지 -> 복구
F = np.fft.fft2(lena)   #fft2 : 2차원 이미지용
Mag = np.abs(F)     #폭(크기성분) -> 주로 이 부분을 변경
Mag = np.fft.fftshift(Mag)
Pha = np.angle(F)   #위상 -> 위상은 거의 건들지 않음

#주파수 영역에서 사용하기 위한 필터를 만들기
#하이패스(중간0), 로우패스(중간1), 원 지름 30
#중심부 좌표 설정
row = 256
col = 256
cx = 128
cy = 127
Radius = 30

Ideal_Low = np.zeros(shape=(row, col), dtype=np.uint8)
Ideal_High = np.zeros(shape=(row, col), dtype=np.uint8)

for y in range(row):
    for x in range(col):
        d = np.sqrt((x-cx)**2 + (y-cy)**2)
        if d > Radius :
            Ideal_Low[y, x] = 0
            Ideal_High[y, x] = 1
        else :
            Ideal_Low[y, x] = 1
            Ideal_High[y, x] = 0

#만들어진 필터와 원본 이미지의 Mag와 곱해줌 (필터링)
Mag_Low = Mag*Ideal_Low
Mag_High = Mag*Ideal_High

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

IM_High = np.abs(IM_High)

### 출력 ###
plt.subplot(231)
plt.gray()
plt.axis('off')
plt.title('Original lena Image')
plt.imshow(lena)

plt.subplot(232)
plt.gray()
plt.axis('off')
plt.title('Ideal Low Image')
plt.imshow(Ideal_Low)

plt.subplot(233)
plt.gray()
plt.axis('off')
plt.title('Ideal High Image')
plt.imshow(Ideal_High)

plt.subplot(234)
plt.gray()
plt.axis('off')
plt.title('Ideal Low + lena Image')
plt.imshow(IM_Low)

plt.subplot(235)
plt.gray()
plt.axis('off')
plt.title('Ideal High + lena Image')
plt.imshow(IM_High)

plt.show()