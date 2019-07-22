import numpy as np


#4장 비트 평면 분할에 관련된 코드이다.
#0~255 범위 값을 가지는 픽셀들을 2진수 배열로 만들어 주는 코드이다.
#그러나 0과1로 표현되는 것이 아니라
#각 자리에 대응 되는 [128, 64, 32...,2, 1]로 표현이 된다.
#특징으로는 맨 앞의 128은 128개의 정보를 담고 있다고도 볼수 있다.
#맨 마지막 1의 경우엔 0과1의 2가지 정보 뿐이다.
#거의 정보가 담겨 있지 않아, 이 맨 마지막 비트자리에 워터 마크를 심어 줄수도 있다.

#100를 출력할 경우
#[0. 64. 32.  0.  0.  4.  0.  0.]
#이라는 결과가 나온다.

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

value = 100
Bits_Value = BitPlane_Slice(value)
print(Bits_Value)