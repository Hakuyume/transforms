#! /usr/bin/env python3

import argparse
import cv2

from transforms import fb_resnet_torch


if __name__ == '__main__':
    modes = {
        'fb_resnet_torch.imagenet':
        lambda x: fb_resnet_torch.imagenet.preprocess(x, train=True),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('--mode', choices=list(modes.keys()))
    args = parser.parse_args()

    if args.mode in modes:
        preprocess = modes[args.mode]
    else:
        def preprocess(x):
            return x

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    while True:
        cv2.imshow('sample', preprocess(image) / 255)
        if cv2.waitKey() == ord('q'):
            break
