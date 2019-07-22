import numpy as np
import matplotlib.pyplot as plt

num_samples = 100

# The sampling rate of the analog to digital convert
sampling_rate = 100.0
amplitude = 1

#w0 = [0, 1/8, 1/4, 1/2, 1, 3/2, 7/4, 15/8, 2, 2.125, 2.25, 2.5, 3, 3.5, 3.75,  3.875, 4, 4.125]
f0 = [0, 1/8, 1/4, 1/2, 1, 3/2, 7/4, 15/8, 2]
freq_num = len(f0)
n = np.arange(num_samples)

for i in range(freq_num):
    x = amplitude*np.cos(f0[i]*np.pi*n)
    plt.subplot(3,3,i+1), plt.stem(x), plt.title('Cosine Waveform : %f' %f0[i])

plt.show()



