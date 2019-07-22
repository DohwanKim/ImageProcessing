import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, misc

#이미지의 주파수, 위상을 이미지 파일로 추출

# Image Read
lena = misc.imread('./image_sample/lena_256.bmp')
lena_noisy = misc.imread('./image_result/lena_noisy.bmp')
lena_Mag_noise = misc.imread('./image_result/lena_Mag_noise.bmp')
col, row = lena.shape

#버터워스 필터
col = 256
row = 256
cx = 160
cy = 166
D0 = 50 #필터 지름
N = 1 #페이딩 강도
W = 20 #필터 두께

BW_L = np.zeros(shape=[col,row], dtype=np.float32)
BW_H = np.zeros(shape=[col,row], dtype=np.float32)
BW_BP = np.zeros(shape=[col,row], dtype=np.float32)
BW_BS = np.zeros(shape=[col,row], dtype=np.float32)

for x in range(col):
    for y in range(row):
        D = np.sqrt((cx-x)**2 + (cy-y)**2)
        s = (W * D) / (D ** 2 - D0 ** 2)
        BW_L[x, y] = 1 / (1 + pow(D/D0, 2*N))
        BW_H[x, y] = 1 / (1 + pow(D0/D, 2*N))
        BW_BS[x, y] = 1 / (1 + pow(s, 2*N))

BW_BP = 1 - BW_BS

# Apply Fourier TransForm
# 이미지 푸리에 변환 -> 주파수 영역 이미지 -> 복구
F = np.fft.fft2(lena)   #fft2 : 2차원 이미지용
F_n = np.fft.fft2(lena_noisy)
Mag = np.abs(F)     #폭(크기성분) -> 주로 이 부분을 변경
Mag_n = np.abs(F_n)
Mag = np.fft.fftshift(Mag)
Mag_n = np.fft.fftshift(Mag_n)
Pha = np.angle(F)   #위상 -> 위상은 거의 건들지 않음
Pha_n = np.angle(Mag_n)

#주파수와 위상를 이미지 파일로 저장
# misc.imsave('./image_result/lena_Mag.bmp', Mag)
# misc.imsave('./image_result/lena_Pha.bmp', Pha)

Used_Filter = BW_L
Mag_BP = lena_Mag_noise*Used_Filter

Mag_noise = np.fft.fftshift(lena_Mag_noise)
Mag_noise_recover = np.fft.fftshift(Mag_BP)

Complex_noise = Mag_noise*np.exp(1j*Pha)
Complex_noise_recover = Mag_noise_recover*np.exp(1j*Pha)

IM_noise = np.fft.ifft2(Complex_noise)
IM_noise = np.real(IM_noise)

IM_noise_recover = np.fft.ifft2(Complex_noise_recover)
IM_noise_recover = np.real(IM_noise_recover)

plt.subplot(331)
plt.imshow(lena)
plt.gray()
plt.axis('off')
plt.title('Original Image')

plt.subplot(332)
plt.imshow(np.log10(Mag)+1)
plt.gray()
plt.axis('off')
plt.title('Mag(log10(Mag)+1)')

plt.subplot(333)
plt.imshow(Pha)
plt.gray()
plt.axis('off')
plt.title('Pha')

plt.subplot(334)
plt.imshow(Mag_noise)
plt.gray()
plt.axis('off')
plt.title('Noise Mag')

plt.subplot(335)
plt.imshow(IM_noise)
plt.gray()
plt.axis('off')
plt.title('lena with noise mag')

plt.subplot(336)
plt.imshow(Used_Filter)
plt.gray()
plt.axis('off')
plt.title('Used filter')

plt.subplot(337)
plt.imshow(Mag_BP)
plt.gray()
plt.axis('off')
plt.title('noise mag + bandpass filter')

plt.subplot(339)
plt.imshow(IM_noise_recover)
plt.gray()
plt.axis('off')
plt.title('Recover with noise mag lena')

plt.show()


plt.subplot(221)
plt.imshow(lena_noisy)
plt.gray()
plt.axis('off')
plt.title('noisy lena')


plt.subplot(222)
plt.imshow(np.log10(Mag_n+1))
plt.gray()
plt.axis('off')
plt.title('noisy lena mag')

plt.subplot(223)
plt.imshow(Pha_n)
plt.gray()
plt.axis('off')
plt.title('noisy lena pha')

plt.show()