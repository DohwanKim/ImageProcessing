import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, misc
from scipy import ndimage

def Im_filtering(im, Filter, FilterSize, dummyNum):
    row, col = im.shape
    Padding = int(FilterSize/2)
    Image_Buffer = np.zeros(shape=(row+2*Padding, col+2*Padding), dtype=np.uint8)
    Image_Buffer[Padding:row+Padding, Padding:col+Padding] = im[:, :]
    Image_New = np.zeros(shape=(row, col), dtype=np.uint8)

    for y in range(Padding, row+Padding):
        for x in range(Padding, col+Padding):
            buff = Image_Buffer[y-Padding:y+Padding+1, x-Padding:x+Padding+1]
            pixel = np.sum(buff * Filter) + dummyNum
            pixel = np.uint8(np.where(pixel>255, 255, np.where(pixel<0, 0, pixel)))
            Image_New[y-Padding, x-Padding] = pixel
    return Image_New

def Median_filtering(im, FilterSize):
    row, col = im.shape
    Padding = int(FilterSize/2)

    Image_Buffer = np.zeros(shape=(row+2*Padding, col+2*Padding), dtype = np.uint8) # g
    Image_Buffer[Padding:row+Padding, Padding:col+Padding] = im[:, :]
    Image_New = np.zeros(shape=(row,col), dtype = np.uint8)

    for y in range(Padding,row+Padding):
        for x in range(Padding,col+Padding):
            buff = Image_Buffer[y-Padding:y+Padding+1, x-Padding:x+Padding+1]
            pixel = np.sort(buff, axis=None)[4]
            Image_New[y-Padding, x-Padding] = pixel

    return Image_New

def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row, col = image.shape
        mean = 0
        var = 0.5 #가우시안 정도
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = image + 10.0 * gauss
        return noisy

    elif noise_typ == "s&p":
        row, col = image.shape
        s_vs_p = 0.5
        amount = 0.05 #소금 후추 효과 정
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 255
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * 10.0*vals) / float(vals)
        return noisy

    elif noise_typ =="speckle":
        row,col = image.shape
        gauss = np.random.randn(row,col)
        gauss = gauss.reshape(row,col)
        noisy = image + image * 0.1*gauss
        return noisy

# Image Read
lena = misc.imread('./image_sample/lena_256.bmp')
col, row = lena.shape
IM_gauss = noisy("gauss", lena)
IM_sp = noisy("s&p", lena)
IM_poisson = noisy("poisson", lena)
IM_speckle = noisy("speckle", lena)

#Fileter03 blur (value average)
meanblur_FilterSize = 11    #필터 사이즈가 커질수록 더 흐려짐
meanblurFilter = np.ones(shape=(meanblur_FilterSize, meanblur_FilterSize))  #정해준 필터 크기 만큼 행렬을 만들고 각 값을 1로 만들어줌
meanblurFilter = meanblurFilter/np.sum(meanblurFilter) #필터 행렬 인덱스 갯수를 분수로 해서 모든 인덱스를 나눠준다. (여기서는 1/25)
IM_mean = Im_filtering(IM_sp, meanblurFilter, meanblur_FilterSize, 0)

#mediannFilter
medianFilterSize = 3
IM_sp_recover = Median_filtering(IM_sp, medianFilterSize)


plt.subplot(231)
plt.gray()
plt.axis('off')
plt.imshow(lena)
plt.title('Original Image')

plt.subplot(232)
plt.title('Gaussian Image')
plt.imshow(IM_gauss)
plt.gray()
plt.axis('off')

plt.subplot(233)
plt.title('s&p Image(amount:0.05)')
plt.imshow(IM_sp)
plt.gray()
plt.axis('off')

plt.subplot(234)
plt.title('s&p + mean Image(mean:11)')
plt.imshow(IM_mean)
plt.gray()
plt.axis('off')

plt.subplot(235)
plt.title('s&p+mean recover Image (used 3x3 sort)')
plt.imshow(IM_sp_recover)
plt.gray()
plt.axis('off')

plt.show()