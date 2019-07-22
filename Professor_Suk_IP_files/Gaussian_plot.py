import numpy as np
from scipy import signal, misc
import matplotlib.pyplot as plt
from scipy import ndimage

#Method 1
sigma = 10.0
G = np.zeros(shape=(256), dtype = np.float)
for x in range(-127, 128):
    s = 1/(np.sqrt(2*np.pi)*sigma)
    v = -(pow(x,2))/(2*pow(sigma,2))
    G[x+127] = s*np.exp(v)

plt.plot(G)
plt.title('1-D Gaussian')
plt.show()

#Method 2
sigma = 10.0
x_axis = np.arange(-127, 128)
s = 1/(np.sqrt(2*np.pi)*sigma)
v = -(pow(x_axis,2))/(2*pow(sigma,2))
G1 = s*np.exp(v)

plt.plot(G1)
plt.title('1-D Gaussian')
plt.show()

#2-D Gaussian
sigma = 10.0
G = np.zeros(shape=(256,256), dtype = np.float)
for y in range(-127, 128):
    for x in range(-127, 128):
        s = 1/(2*np.pi*pow(sigma,2))
        v = -(pow(x,2)+pow(y,2))/(2*pow(sigma,2))
        G[y+127, x+127] = s*np.exp(v)

plt.imshow(G)
plt.gray()
plt.show()


#method 4
from mpl_toolkits.mplot3d import Axes3D
sigma = 20.0
x = np.arange(-127, 128, 1.0)
y = np.arange(-127, 128, 1.0)
X, Y = np.meshgrid(x, y)
s = 1/(2*np.pi*pow(sigma,2))
v = -(pow(X,2)+pow(Y,2))/(2*pow(sigma,2))
G = s*np.exp(v)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, G)
plt.show()


#2-D Gaussian
sigma = 1.0
FilterSize = 9
G = np.zeros(shape=(FilterSize,FilterSize), dtype = np.float)
for y in range(-4, 5):
    for x in range(-4, 5):
        s = 1/(2*np.pi*pow(sigma,2))
        v = -(pow(x,2)+pow(y,2))/(2*pow(sigma,2))
        G[y+4, x+4] = s*np.exp(v)
        print('{:.4f}' .format(G[y+4, x+4]), end=' ')
    print("")



























