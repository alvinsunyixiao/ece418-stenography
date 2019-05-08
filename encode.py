#!/usr/bin/python3

from scipy.fftpack import dctn, idctn
from struct import pack, unpack
from skimage import io
from config import *

import argparse
import numpy as np
import scipy as scp

def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='path to the main image')
    parser.add_argument('hidden', type=str,
            help='path to the image to be hidden inside the main image')
    parser.add_argument('-o', '--output', type=str, default='encoded.tiff',
            help='path to the output image, default to encoded.tiff')
    parser.add_argument('-r', '--reduce_ratio', type=float, default=0.95,
            help='contrast reduce ratio to prevent from values out of range error,\
                 default to 0.95')
    return parser.parse_args()

def _img_reduce_constast(img, args):
    reduce_offset = (1 - args.reduce_ratio) / 2
    return (img * args.reduce_ratio) + reduce_offset

def _get_hidden_image_string(args):
    with open(args.hidden, 'rb') as f:
        data = f.read()
    return data

def _max_msg_length(imdct, msg, args):
    h,w,c = imdct.shape
    h_max = int(MAX_FREQ*h)
    h_min = int(MIN_FREQ*h)
    w_max = int(MAX_FREQ*w)
    w_min = int(MIN_FREQ*w)

    mask = np.absolute(imdct[h_min:h_max, w_min:w_max, :]) < THRESH

    return mask.sum()

def _encode_length(imdct, msg):
    h, w, c = imdct.shape
    length  = pack('I', len(msg))
    cnt = 0
    i = int(MAX_FREQ*h) + 1
    for j in range(int(MAX_FREQ*w), int(MIN_FREQ*w), -1):
        for k in range(c):
            if cnt >= len(length):
                return
            if abs(imdct[i,j,k]) < THRESH:
                imdct[i,j,k] = np.sign(imdct[i,j,k]) * length[cnt] * SCALE
                cnt += 1

def _encode_image(imdct, msg):
    h, w, c = imdct.shape
    cnt = 0;
    for i in range(int(MAX_FREQ*h), int(MIN_FREQ*h), -1):
        for j in range(int(MAX_FREQ*w), int(MIN_FREQ*w), -1):
            for k in range(c):
                if cnt >= len(msg):
                    return
                if abs(imdct[i,j,k]) < THRESH:
                    imdct[i,j,k] = np.sign(imdct[i,j,k]) * msg[cnt] * SCALE
                    cnt += 1

def main(args):
    # get image data
    img = io.imread(args.image).astype('float32') / 255
    img = _img_reduce_constast(img, args)
    enc_msg = _get_hidden_image_string(args)
    # compute dct transform
    img_dct = dctn(img, axes=[0, 1], norm='ortho')
    # check encoded message length
    max_length = _max_msg_length(img_dct, enc_msg, args)
    assert len(enc_msg) < max_length,\
        'hidden image too large, needs to be resized under {}KB'.format(max_length >> 10)
    # encode computation
    _encode_length(img_dct, enc_msg)
    _encode_image(img_dct, enc_msg)
    # compute inverse dct transform
    img_ret = idctn(img_dct, axes=[0, 1], norm='ortho')
    if img_ret.min() < 0 or img_ret.max() > 1:
        print('Warning: data encoded may not be perfectly recovered or not at all. Consider lowering reduce ratio')
        img_ret = np.clip(img_ret, 0, 1)
    # quantization
    img_ret = (img_ret * QUANT_SCALE).astype('uint16')
    # save to disk
    assert args.output.endswith('.tiff'), 'output image has to be in tiff format'
    io.imsave(args.output, img_ret)


if __name__ == '__main__':
    args = _get_args()
    main(args)

