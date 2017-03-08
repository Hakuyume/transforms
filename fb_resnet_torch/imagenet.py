import cv2
import numpy as np

from transforms.common import (
    center_crop,
    color_jitter,
    horizontal_flip,
    lighting,
)


def scale(image, size):
    h, w, *_ = image.shape

    if min(h, w) == size:
        return image

    if w < h:
        return cv2.resize(image, (size, size * h // w))
    else:
        return cv2.resize(image, (size * w // h, size))


def random_sized_crop(image, size):
    h, w, *_ = image.shape

    for _ in range(9):
        area = h * w * np.random.uniform(0.08, 1.0)
        aspect_ratio = np.random.uniform(3 / 4, 4 / 3)
        crop_h = int(np.sqrt(area / aspect_ratio))
        crop_w = int(np.sqrt(area * aspect_ratio))

        if np.random.rand() < 0.5:
            crop_h, crop_w = crop_w, crop_h

        if crop_h <= h and crop_w <= w:
            top = np.random.randint(h - crop_h + 1)
            left = np.random.randint(w - crop_w + 1)
            image = image[top:top + crop_h, left:left + crop_w]
            return cv2.resize(image, (size, size))

    return center_crop(scale(image, size), size)


_eigval = np.array((0.2175, 0.0188, 0.0045))
_eigvec = np.array((
    (-0.5675, 0.7192, 0.4009),
    (-0.5808, -0.0045, -0.8140),
    (-0.5836, -0.6948, 0.4203),
)).transpose()[::-1]


def preprocess(image, size=224, train=False, mean=False):
    image = image.astype(float)

    if train:
        image = random_sized_crop(image, size)
        image = color_jitter(
            image,
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        )
        image = lighting(image, 255 * 0.1, _eigval, _eigvec)
        image = horizontal_flip(image, 0.5)
    else:
        image = scale(image, 256)
        image = center_crop(image, size)

    if mean:
        image -= (103.063, 115.903, 123.152)

    return image
