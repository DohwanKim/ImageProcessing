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
lena = misc.imread('./image_sample/DongRyn.bmp')
row, col = lena.shape

#Filter01 Identitly -> 자기 자신이 나오는 필터
Identity_FilterSize = 5
Identity_Filter = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]])

#Filter02 Embossing
Embossing_FilterSize = 3
sum_dummy = 128
# Embossing_Filter = np.array([
#     [-1, 0, 0],
#     [0, 0, 0],
#     [0, 0, 1]])

#깊이를 기존 엠보싱 필터에서 반전
Embossing_Filter = np.array([
    [1, 0, 0],
    [0, 0, 0],
    [0, 0, -1]])

#Fileter03 blur (value average)
meanblur_FilterSize = 5    #필터 사이즈가 커질수록 더 흐려짐
meanblurFilter = np.ones(shape=(meanblur_FilterSize, meanblur_FilterSize))  #정해준 필터 크기 만큼 행렬을 만들고 각 값을 1로 만들어줌
meanblurFilter = meanblurFilter/np.sum(meanblurFilter) #필터 행렬 인덱스 갯수를 분수로 해서 모든 인덱스를 나눠준다. (여기서는 1/25)

#Fileter03 blur_02 (more big filter)
meanblur_FilterSize02 = 11
meanblurFilter02 = np.ones(shape=(meanblur_FilterSize02, meanblur_FilterSize02))
meanblurFilter02 = meanblurFilter02/np.sum(meanblurFilter02)

#Fileter04 weighted mean filter
Weighted_FilterSize = 3
Weighted_Filter = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]])

Weighted_Filter = Weighted_Filter/np.sum(Weighted_Filter)



#result
Image_Embossing = Im_filtering(lena, Embossing_Filter, Embossing_FilterSize, sum_dummy)
Image_Identity = Im_filtering(lena, Identity_Filter, Identity_FilterSize, 0)
Image_meanblur = Im_filtering(lena, meanblurFilter, meanblur_FilterSize, 0)
Image_meanblur02 = Im_filtering(lena, meanblurFilter02, meanblur_FilterSize02, 0)
Image_Weighted = Im_filtering(lena, Weighted_Filter, Weighted_FilterSize, 0)

# ---영규 코드
# for y in range(1, row-1):
#     for x in range(1, col-1):
#         Image_Buffer[y][x] = np.sum(lena[y-1:y+2, x-1:x+2] * Identity_Filter)
# 영규 코드 끝---


plt.subplot(2,3,1)
plt.gray()
plt.axis('off')
plt.imshow(lena)
plt.title('Original Images')

plt.subplot(2,3,2)
plt.gray()
plt.imshow(Image_Identity)
plt.title('Identity Filter Images')

plt.subplot(2,3,3)
plt.gray()
plt.imshow(Image_Embossing)
plt.title('Embossing Filter Images')

plt.subplot(2,3,4)
plt.gray()
plt.imshow(Image_meanblur)
plt.title('blur Filter Images (filter size : 5)')

plt.subplot(2,3,5)
plt.gray()
plt.imshow(Image_meanblur02)
plt.title('blur Filter Images (filter size : 11)')

plt.subplot(2,3,6)
plt.gray()
plt.imshow(Image_Weighted)
plt.title('Weighted Filter (3x3)')


plt.show()
