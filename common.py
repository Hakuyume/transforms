import cv2
import numpy as np


def center_crop(image, size):
    h, w, *_ = image.shape
    top, left = (h - size) // 2, (w - size) // 2
    return image[top:top + size, left:left + size]


def grayscale(image):
    return cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY)


def brightness(image, value):
    alpha = 1 + np.random.uniform(-value, value)
    return image * alpha


def contrast(image, value):
    alpha = 1 + np.random.uniform(-value, value)
    return image * alpha + grayscale(image).mean() * (1 - alpha)


def saturation(image, value):
    alpha = 1 + np.random.uniform(-value, value)
    return image * alpha + grayscale(image)[:, :, np.newaxis] * (1 - alpha)


def color_jitter(image, **kwargs):
    ts = list()

    if 'brightness' in kwargs:
        ts.append(lambda image: brightness(image, kwargs['brightness']))
    if 'contrast' in kwargs:
        ts.append(lambda image: contrast(image, kwargs['contrast']))
    if 'saturation' in kwargs:
        ts.append(lambda image: saturation(image, kwargs['saturation']))

    if len(ts) == 0:
        return image

    for i in np.random.permutation(len(ts)):
        image = ts[i](image)
    return image


def lighting(image, alphastd, eigval, eigvec):
    alpha = np.random.normal(0, alphastd, size=3)
    return image + eigvec.dot(eigval * alpha)


def horizontal_flip(image, ratio):
    if np.random.rand() < ratio:
        return image[:, ::-1]
    return image
