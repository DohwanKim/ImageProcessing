import numpy as np

class Im_Filtering:
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