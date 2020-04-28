from utilities.threshold import threshold_phansalkar

import matplotlib
import PyQt5
import matplotlib.pyplot as plt

from skimage.filters import threshold_multiotsu, threshold_sauvola
from skimage import io, color, exposure

from scipy import ndimage
import numpy as np
import cv2
matplotlib.use('Qt5Agg')


def main():
    image = io.imread("~/Pictures/Kaddy Geschenk/20190729_115340.jpg")
    gray = color.rgb2gray(image)
    crop = gray[1300:2300, 2600:3600]
    # crop = gray[2200:2300, 3500:3600]

    # processing pipeline
    # img_adapteq = exposure.equalize_adapthist(crop, clip_limit=0.03)
    img_adapteq = exposure.equalize_hist(crop)
    img_denoise = ndimage.gaussian_filter(img_adapteq, 0)
    # thresholds = threshold_multiotsu(img_denoise, classes=10)
    # regions = np.digitize(crop, bins=thresholds)
    # regions = threshold_sauvola(img_denoise)
    regions = threshold_phansalkar(crop, 50)

    plt.imshow(regions, cmap='gray')
    cv2.imwrite('foo.png', regions*255)
    plt.show()


if __name__ == '__main__':
    main()
