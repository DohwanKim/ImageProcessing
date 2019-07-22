import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, misc
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage

#시그마값이 커지면, 가우시안의 높이는 낮지만 폭이 넓어지게 된다.
#즉, 시그마의 값이 커지게 되면, 블러링 되는 정도도 커지게 된다.

def Im_filtering(im, Filter, FilterSize, dummyNum):
    #이미지의 형태 불러오기
    row, col = im.shape

    #필터사이즈의 절반을 취해 페딩 사이즈를 설정 (이하 주석엔 페딩 2를 기준으로 설명)
    Padding = int(FilterSize/2)

    #페딩 사이즈를 취해준 새로운 배열 생성(255+(2x2)=259, 259)
    Image_Buffer = np.zeros(shape=(row+2*Padding, col+2*Padding), dtype=np.uint8)

    #페딩을 해준 이미지를 만듬 -> [259, 259]에 [2,2]부터 이미지를 채워넣음.
    #이렇게 해주면 결국 원본 이미지는 중간에 있게 됨.
    Image_Buffer[Padding:row+Padding, Padding:col+Padding] = im[:, :]

    #필터가 적용된 255,255 사이즈 이미지를 담아줄 그릇
    Image_New = np.zeros(shape=(row, col), dtype=np.uint8)

    #각 2~257까지 돌아가는 중첩 for문 2개 (255번)
    for y in range(Padding, row+Padding):
        for x in range(Padding, col+Padding):
            #버퍼를 만들어준다
            #[(0~259), (0~259)] (뒷 인자는 260을 적어놔야 259까지 작동)
            buff = Image_Buffer[y-Padding:y+Padding+1, x-Padding:x+Padding+1]
            #버퍼와 필터를 곱한 값의 배열을 전부 더해서 픽셀에 담아준다.
            pixel = np.sum(buff * Filter) + dummyNum
            #오버플로우를 방지한다
            pixel = np.uint8(np.where(pixel>255, 255, np.where(pixel<0, 0, pixel)))
            #마지막으로 이미지에 0~255, 0~255에 담아준다.
            Image_New[y-Padding, x-Padding] = pixel
    return Image_New

#sample source
lena = misc.imread('./image_sample/lena_256.bmp')
row, col = lena.shape

sigma = 30.0

#for문을 사용한 2차원 가우시안
G = np.zeros(shape=(256, 256), dtype=np.float)
for x in range(-127, 128):
    for y in range(-127, 128):
        #가우시안 각 항을 나눠서 써줌
        s1 = -(1/np.pi * pow(sigma, 4))
        s2 = 1-((pow(x, 2)+pow(y, 2))/(2 * pow(sigma, 2)))
        v = -(pow(x,2)+pow(y,2))/(2*pow(sigma, 2))
        G[y+127, x+127] = s1*s2*np.exp(v)

#만들어진 데이터는 256x256 이미지 파일로 표현됨
#(a)
plt.imshow(G)
plt.title('LoG Filter, sigma:'+str(sigma))
plt.gray()
plt.show()

#3차원
#(b)
x = np.arange(-127,128, 1.0)
y = np.arange(-127,128, 1.0)
x1, y1 = np.meshgrid(x, y)
s1 = -(1 / np.pi * pow(sigma, 4))
s2 = 1 - ((pow(x1, 2) + pow(y1, 2)) / (2 * pow(sigma, 2)))
v = -(pow(x1, 2)+pow(y1, 2)) / (2 * pow(sigma, 2))
g_3d = s1*s2*np.exp(v)
fig = plt.figure()

ax = Axes3D(fig)
ax.plot_surface(x1, y1, g_3d)
plt.show()

#9x9 LoG 마스크
sigma = 0.8
G = np.zeros(shape=(9, 9), dtype=np.float)
for x in range(-4, 5):
    for y in range(-4, 5):
        #가우시안 각 항을 나눠서 써줌
        s1 = -(1/np.pi * pow(sigma, 4))
        s2 = 1-((pow(x, 2)+pow(y, 2))/(2 * pow(sigma, 2)))
        v = -(pow(x,2)+pow(y,2))/(2*pow(sigma, 2))
        G[y+4, x+4] = s1*s2*np.exp(v)

#(c)
plt.imshow(G)
plt.title('LoG Filter, sigma:'+str(sigma))
plt.gray()
plt.show()

#(d)
Image_LoD = Im_filtering(lena, G, 9, 0)

plt.subplot(1,2,1)
plt.gray()
plt.axis('off')
plt.imshow(lena)
plt.title('Original Images')

plt.subplot(1,2,2)
plt.gray()
plt.axis('off')
plt.imshow(Image_LoD)
plt.title('LoD Images (FilterSize9, sigma:0.8)')

plt.show()

#(e)
#어느 한 픽셀과 인접한 다른 픽셀의 값의 변화가 클 경우
#이 필터는 선을 끄어준다.
#변화가 크면 클수록 밝은 값을 넣어 입체감도 느낄수 있다
#결과적으로 일종의 경계선 처리 필터이다

