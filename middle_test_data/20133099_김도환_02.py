import numpy as np
from scipy import signal, misc
import matplotlib.pyplot as plt
from scipy import ndimage

#필터 함수 (원본 데이터, 필터, 필터 사이즈, 결과에 취해 줄 정수)
#경계처리 적용 (패딩 기법)
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

#Fileter blur (value average = mean)
meanblur_FilterSize = 3
meanblurFilter = np.ones(shape=(meanblur_FilterSize, meanblur_FilterSize))
meanblurFilter = meanblurFilter/np.sum(meanblurFilter)
#필터링 함수 적용
meanblurFilter = Im_filtering(lena, meanblurFilter, meanblur_FilterSize, 0)

#수식 적용하고 변수에 담아줌. 오버플로우 처리를 위해 32비트 실수화
Filter_result = np.float32((lena + (lena - meanblurFilter)))

#오버 플로우 처리
Filter_result = np.where(Filter_result>255, 255, np.where(Filter_result<0, 0, Filter_result))

#출력
plt.subplot(121)
plt.imshow(lena)
plt.title('Original Image')
plt.gray()
plt.axis('off')

plt.subplot(122)
plt.imshow(Filter_result)
plt.title('Filtering Image')
plt.gray()
plt.axis('off')
plt.show()

#(b)
#간단한 선명도 향상 필터로 보입니다. -> 포토샵에선 샤픈 필터
#블러가 적용된 흐려진 레나 이미지에서 빼줌으로써 선명도를 위한 더미가 만들어집니다.
#이것을 원본에 더해버리면 결과적으로 선명해 집니다.
#포토샵에서 샤픈값을 과하게 주면 픽셀이 깨져 값을 낮출 때가 있는데
#이 필터를 적용한 이미지에서도 그러한 현상이 일어나는 것 같습니다.

