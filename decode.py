#!/usr/bin/python3

from scipy.fftpack import dctn, idctn
from struct import pack, unpack
from io import BytesIO
from skimage import io
from PIL import Image
from config import *

import argparse
import numpy as np
import scipy as scp

def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='path to the main image')
    parser.add_argument('-o', '--output', type=str, default='decoded.png',
            help='path to the output image, default to decoded.tiff')
    return parser.parse_args()

def _decode_length(imdct):
    h, w, c = imdct.shape
    length = b''
    i = int(MAX_FREQ*h) + 1
    for j in range(int(MAX_FREQ*w), int(MIN_FREQ*w), -1):
        for k in range(c):
            if len(length) == 4:
                return unpack('I', length)[0]
            if abs(imdct[i,j,k]) < THRESH:
                length += pack('B', int(np.round(abs(imdct[i,j,k])/SCALE)))

def _decode_image(imdct, length):
    h, w, c = imdct.shape
    msg = b''
    cnt = 0
    for i in range(int(MAX_FREQ*h), int(MIN_FREQ*h), -1):
        for j in range(int(MAX_FREQ*w), int(MIN_FREQ*w), -1):
            for k in range(c):
                if len(msg) == length:
                    return msg
                if abs(imdct[i,j,k]) < THRESH:
                    msg += pack('B', int(np.round(abs(imdct[i,j,k])/SCALE)))


def main(args):
    # read in encoded image
    img = io.imread(args.image)
    assert img.dtype == np.dtype('uint16'), 'Cannot decode anything other than 16-bit images'
    img = img.astype('float32') / QUANT_SCALE
    # compute dct transform
    img_dct = dctn(img, axes=[0, 1], norm='ortho')
    # try to decode
    length = _decode_length(img_dct)
    msg = _decode_image(img_dct, length)
    # check validity
    assert msg is not None and len(msg) == length, 'Decode Failed'
    # save to file
    pil_img = Image.open(BytesIO(msg))
    pil_img.save(args.output)


if __name__ == '__main__':
    args = _get_args()
    main(args)

