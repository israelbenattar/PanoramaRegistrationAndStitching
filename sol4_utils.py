import imageio
import skimage.color
from scipy import signal, ndimage
from scipy.signal import convolve2d
import numpy as np


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


def read_image(filename, representation):
    """
    This function reads an image file and converts it into a given representation.
    :param filename: The filename of an image on disk.
    :param representation: Representation code, either 1 or 2 defining whether the output should be a grayscale
                           image (1) or an RGB image (2).
    :return: This function returns an image.
    """
    image = imageio.imread(filename)
    if len(image.shape) == 3 and representation == 1:
        image = skimage.color.rgb2gray(image)
    if image.max() > 1:
        image = np.divide(image, 255)
    return image.astype(np.float64)


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    This function build the gaussian pyramid of the given image.
    :param im: A grayscale image with double values in [0, 1].
    :param max_levels: The maximal number of levels in the resulting pyramid
    :param filter_size: The size of the Gaussian filter
    :return: The resulting pyramid as a standard python array, the filter vector.
    """
    if filter_size == 1:
        filter_vec = np.array([1]).reshape(1, 1)
    else:
        a = np.array([1, 1]).reshape(1, 2)
        filter_vec = a
        while filter_vec.shape[1] < filter_size:
            filter_vec = signal.convolve2d(a, filter_vec)
        filter_vec = filter_vec / filter_vec.sum()
    pry = [im]
    i = 1
    while i < max_levels and pry[-1].shape[0] > 32 and pry[-1].shape[1] > 32:
        temp = ndimage.filters.convolve(pry[-1], filter_vec)
        temp = ndimage.filters.convolve(temp, np.transpose(filter_vec))

        pry.append(temp[::2, ::2])
        i += 1
    return pry, filter_vec
