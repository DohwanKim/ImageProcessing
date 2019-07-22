import numpy as np
from scipy import signal, misc
import matplotlib.pyplot as plt
from scipy import ndimage

#각 픽셀의 값들은 0~255의 값을 가진다. 이는 8비트로 표현 할수 있으며
#픽셀마다 크기 8의 배열을 넣어 배열의 인덱스마다 0과 1로 비트를 표현해주면 된다.
#당연히 길이는 8이다.
#이렇게 각 픽셀마다 비트값이 생성되면 모든 픽셀의 각 배열에서 1~8까지 각 인덱스 위치마다 따로 불러오는
#것도 가능할 것이다. 이렇게 8개의 비트평면 분할(bit-plane)이 된다.
#상위비트: MSB 비트, 최하위 비트: LSB

def BitPlane_Slice(value):
    bin8 = lambda x: ''.join(reversed([str((x >> i) & 1) for i in range(8)]))
    c = np.zeros(8)
    n = 7
    bits = bin8(value)
    for i in range(8):
        p = pow(2, n)
        c[i] = p * int(bits[i])
        n = n - 1
    return c

# value = 100
# Bits_Value = BitPlane_Slice(value)
# print(Bits_Value)

im = misc.imread('./image_sample/DongRyn.bmp') #원본 이미지
im_cp = misc.imread('./image_sample/copyright.bmp') #워터 마크 이미지

row, col = im.shape #레나 이미지 행렬 사이즈 가져옴
Num_BitSlice = 8 #비트 슬라이스의 배열 갯수

#비트평면 분할을 위해 배열 256x256 배열을 생성하고
## 각 인덱스마다 8의 배열을 생성해준다. (3차원)
Image_BitPlane = np.ndarray(shape=(Num_BitSlice, row, col), dtype=np.uint8)


#비트평면을 분할하고 3차원으로 변환하는 for문
for y in range(row):
    for x in range(col):
        value = im[y, x]
        c = BitPlane_Slice(value)   #비트평면 분할 함수 적용

        #비트평면 분할 된 것을 3차원 배열에 담아준다.
        #픽셀당 8개의 인덱스이니 for문은 8번 돌아간다.
        for i in range(Num_BitSlice):
            Image_BitPlane[i, y, x] = c[i]

#이후 마지막 LSB공간에 워터마크 이미지를 넣어준다
Image_BitPlane[7, :, :] = im_cp

#복구 될 이미지를 위한 배열
Image_Restore = np.zeros(shape=(row, col), dtype=np.uint8)

#비트 분할의 인덱스 갯수 만큼 for를 돌리고 출력한다.
#동시에 각 값을 더해 2차원으로 복구한다.
for i in range(Num_BitSlice):
    #0~7자리 마다 각 성분
    img = Image_BitPlane[i, :, :]
    print(img)
    Image_Restore = img + Image_Restore
    plt.subplot(2, 4, i+1)
    plt.imshow(img)
    plt.gray()
    plt.title('BitSlice Image: %i' % (i + 1))
    plt.axis('off')
plt.show()


plt.subplot(1,2,1)
plt.imshow(im)
plt.title('Original Images')

plt.subplot(1,2,2)
plt.imshow(Image_Restore)
plt.title('Restore Images')
plt.show()

#워터마크를 삽입하고 복원한 이미지 파일
misc.imsave('./image_result/DongRyn_w.png', Image_Restore)

