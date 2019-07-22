import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

#!! 과제로 받은 비트 플랜 슬라이스 코드를 한줄로 줄인 코드
# bin8 함수 포함 되어 있음
# bin16도 대응 가능
bit_plane_slice = lambda x : np.array([pow(2, i) * (x >> i & 1) for i in range(7, -1, -1)])

# [pow(2, i) * (x >> i & 1) for i in range(7, -1, -1)]
# range(7, -1, -1)은 7부터 0까지 -1 씩 내려간다. reverse 먹이기 실어서 꺼꾸로 돌림
# (x >> i & 1) 기존 코드에도 있었던 i번째 있는 한비트 뽑아내는 코드이다.
# 그 비트를 다이렉트로 뽑아 내고 pow(2, i) 곱하면 이전 그런 긴 코드 필요 없다.
# 장점. 문자열로 변환 두번씩이나 안해서 존나 빠름 빠름

lena = misc.imread('./lena_256.bmp')
copyright = misc.imread('./copyright.bmp')

#!! 워터 마크 LSB에 삽입하는 한줄 코드
watermark_img = np.where(copyright & 1 == 1,  lena | 1, lena & 254)

# 사실 비트플랜 슬라이스 함수 짯지만 사실 필요 없다.
# np.where은 이미지를 아주 빠르게 순환되서 기존 코드에 딜레이 1도 없다.

# copyright & 1 == 1, 워터마크 이미지 LSB가 1이면!
# lena | 1, 레나 이미지 LSB에 1을 덮어 쓴다. ???????? | 00000001(1) 아니면,
# lena & 254, 레나 이미지 LSB에 0을 덮어 쓴다. ???????? & 11111110(254)
# 매우 간단하다.

#!! 워터마크 추출 함수
img_bit = lambda img, i : np.where((img >> i & 1) == 1, pow(2, i), 0)

# 인자 i에는 몇번째 비트를 추출할껀지 넣어주면 된다.
# ex) LSB 뽑고 십으면 0번째 비트닌까 0 넣으주면 된다.

# 역시 where 써서 빠르다.
# (img >> i & 1) == 1, 이미지의 i번째 비트가 1이면
# pow(2, i) 반환하고, 아니면 0을 뱉어낸다.

# pow(2, i) 불필요하게 반복적으로 연산될것이 걱정되니 다 과 같이 직접 계산해서
# 아래 p 인자처럼 넣어 주면 몸에 좋다.
img_bit = lambda img, i, p : np.where((img >> i & 1) == 1, p, 0)

# 이렇게 반복하면 비트별로 이미지 생성이 간단해지고 덜 느려서 좋다.
for i in range(8): img_bit(watermark_img, i, pow(2, i))


# 마지막으로 이미지 매트 플로 여러개 하는거 힘드닌까 다음 함수를 작성하는 걸 추천한다.
_i = 0
_span = (3, 4)

def setimsnap(snap, i):
    global _i
    global _span

def imsnap(img, title):
    global _i
    global _span
    y, x = _span
    _i += 1
    plt.subplot(y, x, _i)
    plt.imshow(img)
    plt.gray()
    plt.axis('off')
    plt.title(title)

# 이거만 있으면 이미지 여러개 출력하는 부담이 매우 줄어든다.
for i in range(8): imsnap(img_bit(watermark_img, i, pow(2, i)), 'img' + i )