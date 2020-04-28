import numpy as np


def threshold_phansalkar_multi(image, max_radius, classes=3):

    result = np.zeros(image.shape)
    arr = (3, 7, 15)
    for radius in arr:
        result += threshold_phansalkar(image, radius)

    return result


def threshold_phansalkar(img, radius):
    """
    https://landscapearchaeology.org/2018/numpy-loops/
    :param img:
    :param radius:
    :return:
    """
    # TODO check if image is grayscale

    size = radius // 2  # window size i.e. here is 3x3 window

    img = np.pad(img, size, 'edge')
    img_shape = img.shape

    shape = (img.shape[0] - size + 1, img.shape[1] - size + 1, size, size)
    strides = 2 * img.strides
    patches = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    patches = patches.reshape(-1, size, size)

    output_img = np.array([_phansalkar(roi) for roi in patches])
    output_img = np.reshape(output_img, shape[:2])

    return output_img


def _phansalkar(arr, k=0.25, r=0.5):
    """
    This is a modification of Sauvola's thresholding method to deal with low contrast images.

        Phansalskar, N; More, S & Sabale, A et al. (2011),
        "Adaptive local thresholding for detection of nuclei in diversity stained cytology images.",
        International Conference on Communications and Signal Processing (ICCSP): 218-220,
        doi:10.1109/ICCSP.2011.5739305

    In this method, the threshold t is computed as:

    t = mean * (1 + p * exp(-q * mean) + k * ((stdev / r) - 1))

    where mean and stdev are the local mean and standard deviation respectively. Phansalkar recommends k = 0.25,
    r = 0.5, p = 2 and q = 10. In this plugin, k and r are the parameters 1 and 2 respectively, but the values of p and
    q are fixed.

    :param arr: Input window for the threshold.
    :param k: The default value is 0.25. Any other number than 0 will change its value.
    :param r: The default value is 0.5. This value is different from Sauvola's because it uses the normalised intensity
        of the image. Any other number than 0 will change its value.
    :return: 1 if the center if higher then the threshold.
    """
    p = 2
    q = 10

    mean = np.mean(arr)
    stdev = np.std(arr)

    center = np.take(arr, arr.size // 2)

    t = mean * (1 + p * np.exp(-q * mean) + k * ((stdev / r) - 1))

    return 1 if center > t else 0
