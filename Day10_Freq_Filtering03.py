import numpy as np
from scipy import signal, misc
import matplotlib.pyplot as plt
from scipy import ndimage

#시험은 잡음이 낀 신호를 주고 제거하는 식으로 한다.

# Image Read
lena = misc.imread('./image_sample/lena_256.bmp')
col, row = lena.shape

# Apply Fourier TransForm
# 이미지 푸리에 변환 -> 주파수 영역 이미지 -> 복구
F = np.fft.fft2(lena)   #fft2 : 2차원 이미지용
Mag = np.abs(F)     #폭(크기성분) -> 주로 이 부분을 변경
Mag = np.fft.fftshift(Mag)
Pha = np.angle(F)   #위상 -> 위상은 거의 건들지 않음


#버터워스 필터
col = 256
row = 256
cx = 128
cy = 128
D0 = 50
N = 3
W = 20

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

#만들어진 필터와 원본 이미지의 Mag와 곱해줌 (필터링)
Mag_L = Mag*BW_L
Mag_H = Mag*BW_H
Mag_BS = Mag*BW_BS
Mag_BP = Mag*BW_BP

#다시 쉬프트
Mag_L = np.fft.fftshift(Mag_L)
Mag_H = np.fft.fftshift(Mag_H)
Mag_BS = np.fft.fftshift(Mag_BS)
Mag_BP = np.fft.fftshift(Mag_BP)


#새로 만들어진 Mag로 복구
Complex_L = Mag_L*np.exp(1j*Pha)
Complex_H = Mag_H*np.exp(1j*Pha)
Complex_BS = Mag_BS*np.exp(1j*Pha)
Complex_BP = Mag_BP*np.exp(1j*Pha)

IM_L = np.fft.ifft2(Complex_L)
IM_H = np.fft.ifft2(Complex_H)
IM_BS = np.fft.ifft2(Complex_BS)
IM_BP = np.fft.ifft2(Complex_BP)

IM_L = np.real(IM_L)
IM_H = np.real(IM_H)
IM_BS = np.real(IM_BS)
IM_BP = np.real(IM_BP)

IM_H = np.abs(IM_H)
IM_BS = np.abs(IM_BS)


#출력
plt.subplot(221), plt.imshow(BW_L), plt.gray(), plt.axis('off'), plt.title('Butterworth Lowpass')
plt.subplot(222), plt.imshow(BW_H), plt.gray(), plt.axis('off'), plt.title('Butterworth Highpass')
plt.subplot(223), plt.imshow(BW_BP), plt.gray(), plt.axis('off'), plt.title('Butterworth Bandpass')
plt.subplot(224), plt.imshow(BW_BS), plt.gray(), plt.axis('off'), plt.title('Butterworth Bandstop')
plt.show()

plt.subplot(221), plt.imshow(Mag_L), plt.gray(), plt.axis('off'), plt.title('Mag*BW_L')
plt.subplot(222), plt.imshow(Mag_H), plt.gray(), plt.axis('off'), plt.title('Mag*BW_H')
plt.subplot(223), plt.imshow(Mag_BS), plt.gray(), plt.axis('off'), plt.title('Mag*BW_BS')
plt.subplot(224), plt.imshow(Mag_BP), plt.gray(), plt.axis('off'), plt.title('Mag*BW_BP')
plt.show()

plt.subplot(221), plt.imshow(IM_L), plt.gray(), plt.axis('off'), plt.title('Butterworth Lowpass Image')
plt.subplot(222), plt.imshow(IM_H), plt.gray(), plt.axis('off'), plt.title('Butterworth Highpass Image')
plt.subplot(223), plt.imshow(IM_BS), plt.gray(), plt.axis('off'), plt.title('Butterworth Bandpass Image')
plt.subplot(224), plt.imshow(IM_BP), plt.gray(), plt.axis('off'), plt.title('Butterworth Bandstop Image')
plt.show()


#3차원 출력 예제
from mpl_toolkits.mplot3d import Axes3D
sigma_3d = 20.0

x = np.arange(0,256, 1.0)
y = np.arange(0,256, 1.0)
X, Y = np.meshgrid(x, y)
D = np.sqrt((cx - X) ** 2 + (cy - Y) ** 2)
s = (W * D) / (D ** 2 - D0 ** 2)

G3d_LP = 1 / (1 + pow(D/D0, 2*N))
G3d_HP = 1 / (1 + pow(D0/D, 2*N))
G3d_BP = 1 / (1 + pow(s, 2*N))
G3d_BS = 1 - G3d_BP

fig = plt.figure()
ax = Axes3D(fig)

ax.plot_surface(X, Y, G3d_HP)
plt.show()
