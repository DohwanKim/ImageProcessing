import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

#시그마값이 커지면, 가우시안의 높이는 낮지만 폭이 넓어지게 된다.
#즉, 시그마의 값이 커지게 되면, 블러링 되는 정도도 커지게 된다.

#for문을 사용한 1차원 가우시안
sigma_1d = 10.0
G_1d = np.zeros(shape=256, dtype=np.float32)

#가우시안 1차원식 for문
for x in range(-127, 128):          #256x256 사이즈 필터
    #가우시안 1차원식
    s = 1/(np.sqrt(2*np.pi)*sigma_1d)
    v = -(pow(x,2))/(2*pow(sigma_1d, 2))
    G_1d[x+127] = s*np.exp(v)

#arange를 사용한 1차원 가우시안
sigma_1d_a = 10.0
x_axis = np.arange(-127, 128)
s = 1/(np.sqrt(2*np.pi)*sigma_1d_a)
v = -(pow(x_axis,2))/(2*pow(sigma_1d_a,2))
G_1d_a = s*np.exp(v)

plt.subplot(211)
plt.plot(G_1d)
plt.title('1-D Gaussian')
plt.subplot(212)
plt.title('1-D Gaussain(Arange)')
plt.plot(G_1d_a)
plt.show()


#for문을 사용한 2차원 가우시안
sigma_2d = 10.0
G_2d = np.zeros(shape=(256, 256), dtype=np.float)
for x in range(-127, 128):
    for y in range(-127, 128):
        #가우시안 2차원식
        s = 1/(2*np.pi*pow(sigma_2d, 2))
        v = -(pow(x,2)+pow(y,2))/(2*pow(sigma_2d,2))
        G_2d[y+127, x+127] = s*np.exp(v)
#만들어진 데이터는 256x256 이미지 파일로 표현됨
plt.imshow(G_2d)
plt.title('2D Gaussain Filter, sigma:'+str(sigma_2d))
plt.gray()
plt.show()


#3차원 가우시안 출력 예제
from mpl_toolkits.mplot3d import Axes3D
sigma_3d = 20.0

#-127~128까지 1로 나눈 간격으로 배열 생성 -> 여기선 256개 생성
x = np.arange(-127,128, 1.0)
y = np.arange(-127,128, 1.0)
X, Y = np.meshgrid(x, y)
s = 1/(2*np.pi*pow(sigma_3d, 2))
v = -(pow(X,2)+pow(Y,2))/(2*pow(sigma_3d,2))

G_3d = s*np.exp(v)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, G_3d)

plt.show()

