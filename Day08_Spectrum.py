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

plt.subplot(131)
plt.gray()
plt.axis('off')
plt.title('Original Image')
plt.imshow(lena)

plt.subplot(132)
plt.gray()
plt.axis('off')
plt.title('Mag')
plt.imshow(np.log10(Mag)+1) #너무 어두워 값 상향 조정

plt.subplot(133)
plt.gray()
plt.axis('off')
plt.title('Phase')
plt.imshow(Pha)

plt.show()

Mag = np.fft.fftshift(Mag)

#크기성분과 위상만 있으면 주파수 영역 이미지를 복구 가능
MakeComplex = Mag*np.exp(1j*Pha) #=F

#Recover Foruier Image
IM = np.fft.ifft2(F)
IM_MC = np.fft.ifft2(MakeComplex)
IM = np.real(IM) #실수값만 끄집어냄
IM_MC = np.real(IM_MC) #실수값만 끄집어냄


plt.subplot(131)
plt.gray()
plt.axis('off')
plt.title('Original Image')
plt.imshow(lena)

plt.subplot(132)
plt.gray()
plt.axis('off')
plt.title('Recover Image')
plt.imshow(IM)

plt.subplot(133)
plt.gray()
plt.axis('off')
plt.title('MakeComplex Recover Image')
plt.imshow(IM)

plt.show()