from __future__ import print_function
import os
import argparse

import chainer
from chainer import serializers

from model import Generator
from utils import data_process, output2img

import numpy as np

import cv2

from PIL import Image

def main():
    parser = argparse.ArgumentParser(description='pix2pix --- GAN for Image to Image translation')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--load_size', type=int, default=256, help='Scale image to load_size')
    parser.add_argument('--g_filter_num', type=int, default=64, help="# of filters in G's 1st conv layer")
    parser.add_argument('--d_filter_num', type=int, default=64, help="# of filters in D's 1st conv layer")
    parser.add_argument('--output_channel', type=int, default=3, help='# of output image channels')
    parser.add_argument('--n_layers', type=int, default=3, help='# of hidden layers in D')
    parser.add_argument('--list_path', default='list/val_list.txt', help='Path for test list')
    parser.add_argument('--out', default='result/test', help='Directory to output the result')
    parser.add_argument('--G_path', default='result/G.npz', help='Path for pretrained G')
    args = parser.parse_args()

    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    # Set up GAN G
    G = Generator(args.g_filter_num, args.output_channel)
    serializers.load_npz(args.G_path, G)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        G.to_gpu()                               # Copy the model to the GPU

    with open(args.list_path) as f:
        imgs = f.readlines()

    total = len(imgs)
    for idx, img_path in enumerate(imgs):
        print('{}/{} ...'.format(idx+1, total))

        img_path = img_path.strip().split(' ')[-1]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, ::-1]
        h, w, _ = img.shape
        img = np.asarray(Image.fromarray(img).resize((args.load_size, args.load_size), resample=Image.NEAREST), dtype=np.float32)
        img = np.transpose(img, (2, 0, 1))

        A = data_process([img], device=args.gpu, volatile='on')
        B = np.squeeze(output2img(G(A, test=True, dropout=False)))
        
        Image.fromarray(B).resize((w, h), resample=Image.BILINEAR).save(os.path.join(args.out, os.path.basename(img_path).replace('gtFine_labelIds', 'leftImg8bit')))

if __name__ == '__main__':
    main()