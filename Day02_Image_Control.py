import numpy as np
from scipy import signal, misc
import matplotlib.pyplot as plt
from scipy import ndimage

#Image Read
im = misc.imread('./image_sample/DongRyn.bmp')
row, col = im.shape

print(im.shape)
#1. 이미지 반전 (255-원본데이터)
im_inversion = 255 - im

#2,3 이미지 밝기 조절 (원본데이터 + 밝기값)
#원본 데이터를 실수형으로 바꿔둬야 오버플로우 처리가 가능하다.
im_brightness = (np.float32(im))+100 #밝게
im_darkness = (np.float32(im))-100   #어둡게

#오버플로우 처리
im_brightness = np.where(im_brightness>255, 255, im_brightness)
im_darkness = np.where(im_darkness<0, 0, im_darkness)

#4. 명암비 조절
# 알파값이 0~1 사이라면 명암비를 증가시키는 역할을 하게 되고, -1~0 사이의 값을 가지면 명암비를 감소시키는 역할
A = -0.1 #알파값
im_contrast = im + (im - 128)*A

#5. 감마 보정
M = 255
gamma = 0.5
im_gamma = pow(im/M, gamma)*255

#-----응용 파트------
#---1.중간 강조
n = 50      #올릴 값
N1 = 100    #적용할 범위
N2 = 200    #적용할 범위

#원본데이터의 배열을 가진 정수형 변수 생성 -> 결과값을 담아줌
im_middle_ad = np.ndarray(shape=(row, col), dtype=np.uint8)

for y in range(row):
    for x in range(col):
        value = im[y, x]
        #범위 100~200 이라면 n만큼 값을 더해줌
        if value > N1 and value < N2:
            im_middle_ad[y, x] = value + n
        else:
            im_middle_ad[y, x] = value
#0으로 된 (row X col) 배열 생성. (여기서는 255 X 255 이 생성됨)

#---2.이진화
im_binary = np.ndarray(shape=(row, col), dtype=np.uint8)
im_average = np.mean(im) #이미지의 값 평균을 이진화를 위해 지정

for y in range(row):
    for x in range(col):
        value = im[y, x]
        #평균값보다 크면 0, 아니면 255
        if value > im_average:
            im_binary[y, x] = 0
        else:
            im_binary[y, x] = 255

#---3.슬라이스
slice_min = 100
slice_max = 150
slice_bright_value = 50
im_slice = np.ndarray(shape=(row, col), dtype=np.uint8)
for y in range(row):
    for x in range(col):
        value = im[y, x]
        if slice_min > N1 and slice_max < N2:
            im_slice[y, x] = value + slice_bright_value
        else:
            im_slice[y, x] = value

#---4.등명도선
n = 50      #올릴 값
wire_range1_min = 50    #적용할 범위
wire_range1_max = 100    #적용할 범위
wire_range2_min = 150    #적용할 범위
wire_range2_max = 200    #적용할 범위

im_wire = np.ndarray(shape=(row, col), dtype=np.uint8)
for y in range(row):
    for x in range(col):
        value = im[y, x]
        if value > wire_range1_min and value < wire_range1_max:
            im_wire[y, x] = 255
        elif value > wire_range2_min and value < wire_range2_max:
            im_wire[y, x] = 255
        else:
            im_wire[y, x] = 0

#---5.명도단계 변환
a = im
#5단계로 분리해줌
#0~50 값들은 25, 50~100 값들은 75... 씩으로 255까지
#where함수 반복으로 전부다 지
im_brightness_step = np.where(a<50, 25, np.where(a<100, 75, np.where(a<150, 125, np.where(a<200, 175, np.where(a<255, 225, 255)))))

#---6.중간제거와 중간통과
# 중간제거 : 중간부분 명도를 제거
# 중간통과 : 중간부분만 유지 나머지 제거

#적용할 범위 100~150
pass_range_min = 100
pass_range_max = 150

im_pass = np.ndarray(shape=(row, col), dtype=np.uint8)
im_unpass = np.ndarray(shape=(row, col), dtype=np.uint8)

#중간통과 for문
for y in range(row):
    for x in range(col):
        value = im[y, x]
        #100~150 부분만 값을 넣어주고 나머지는 0
        if value > pass_range_min and value < pass_range_max:
            im_pass[y, x] = value
        else:
            im_pass[y, x] = 0

#중간제거
for y in range(row):
    for x in range(col):
        value = im[y, x]
        #100~150 부분은 값을 0으로 하고 나머지는 원본
        if value > pass_range_min and value < pass_range_max:
            im_unpass[y, x] = 0
        else:
            im_unpass[y, x] = value

#--------------------- 출력 ---------------------
#기본 이미지
plt.subplot(2,3,1)
plt.imshow(im)
plt.title('Original Image')
plt.gray()

#반전
plt.subplot(2,3,2)
plt.imshow(im_inversion)
plt.title('Image inversion')
plt.gray()

#밝기 상승
plt.subplot(2,3,3)
plt.imshow(im_brightness)
plt.title('Image brightness')
plt.gray()

#어둡게
plt.subplot(2,3,4)
plt.imshow(im_darkness)
plt.title('Image darkness')
plt.gray()

#명암비 상향
plt.subplot(2,3,5)
plt.imshow(im_contrast)
plt.title('Image contrast')
plt.gray()

#명암비 상향
plt.subplot(2,3,6)
plt.imshow(im_gamma)
plt.title('Image gamma')
plt.gray()

plt.show()

#----응용
#1. 중간 조절
plt.subplot(3,3,1)
plt.imshow(im_middle_ad)
plt.title('Image middle ad')
plt.gray()

#2. 이진화
plt.subplot(3,3,2)
plt.imshow(im_binary)
plt.title('Image binary')
plt.gray()

#3. 슬라이스
plt.subplot(3,3,3)
plt.imshow(im_slice)
plt.title('Image slice')
plt.gray()

#4. 등명도선
plt.subplot(3,3,4)
plt.imshow(im_wire)
plt.title('Image wire')
plt.gray()

#5. 명도 단계 변화
plt.subplot(3,3,5)
plt.imshow(im_brightness_step)
plt.title('Image gamma step')
plt.gray()

#6. 중간제거와 중간통과
plt.subplot(3,3,6)
plt.imshow(im_pass)
plt.title('Image middle pass')
plt.gray()

plt.subplot(3,3,7)
plt.imshow(im_unpass)
plt.title('Image middle delete')
plt.gray()

plt.show()

