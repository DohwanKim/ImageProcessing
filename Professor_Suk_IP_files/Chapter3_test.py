import numpy as np

bin8 = lambda x : ''.join(reversed( [str((x >> i) & 1) for i in range(8)] ) )

def BitPlane_Slice(value):
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