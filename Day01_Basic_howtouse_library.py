# import numpy as np -> numpy 라이브러리
# scipy misc -> 영상을 읽고 쓰는데 필요한 여러 함수를 제공한다. (-> from scipy import signal, misc)
# import matplotlib.pyplot as plt -> 그래픽 관련 라이브러리
# from scipy import ndimage -> 이미지 처리에 필요한 배열생성 지원

from scipy import signal, misc
import matplotlib.pyplot as plt

lena = misc.imread('./image_sample/DongRyn.bmp')   #이미지 읽기
row, col = lena.shape                               #이미지 형태 가져오기
misc.imsave('./image_result/DongRyn_copy.bmp', lena)   #다른이름으로 파일 저장
lena_copy = misc.imread('./image_result/DongRyn_copy.bmp')
row2, col2 = lena_copy.shape

#1
# plt.subplot(321)
# plt.title('Lena Image')             #각 그래프 이름
# plt.axis('off')                     #그래프 데이터 표시 off
# plt.gray()                          #입력 영상이 컬러가 아니므로 흑백으로 지정
# plt.imshow(lena)                    #화면에 영상 출력

#2 for문으로 여러 그래프 표시하기
for i in range(1, 7): #subplot값은 1부터 시작 (1~6)
    plt.subplot(3,2,i)
    plt.title('Lena Image Copy #' + str(i))
    plt.axis('off')
    plt.gray()
    plt.imshow(lena_copy)                    #화면에 영상 출력
plt.show()                          #영상이 사라지지 않고 유지

#subplot()함수
#  화면을 행과 열로 분리하여 여러 개의 그림 출력
#  Ex) plt.subplot(2,2,1) #2행 2열의 4개의 그림 중 첫 번째
#  plt.subplot(2,2,3) #2행 2열의 4개의 그림 중 세 번째
#그림의 설명에 사용되는 함수
#  title() #그림 제목
#  axis() #x-y 축 ON/OFF
#  xlabel(), ylabel() #각각 x축과 y축의 축 설명함수
#  grid() #그림에 grid를 추가하는 함수
#  Ex) plt.title(‘Lena Image’)
#  plt.axis(‘off’)