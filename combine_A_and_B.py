import argparse

import os
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Combine A & B into one image')
    parser.add_argument('--list', type=str, default='list/train_list.txt', help='File path for image list')
    parser.add_argument('--save_dir', type=str, default='datasets/train', help='Folder for storeing combined images')
    args = parser.parse_args()

    with open(args.list) as f:
        img_list = f.readlines()
    
    num_imgs = len(img_list)
    print('{} images in total, storing in {}'.format(num_imgs, args.save_dir))

    img_fold_AB = args.save_dir
    if not os.path.isdir(img_fold_AB):
        os.makedirs(img_fold_AB)
    
    for n in range(num_imgs):
        print('processing {}/{} ...'.format(n, num_imgs))
        
        path_A, path_B = img_list[n].strip().split(' ')

        if os.path.isfile(path_A) and os.path.isfile(path_B):
            name_AB = os.path.basename(path_A)
            path_AB = os.path.join(img_fold_AB, name_AB)
            im_A = cv2.imread(path_A, cv2.IMREAD_COLOR)
            im_B = cv2.imread(path_B, cv2.IMREAD_COLOR)
            im_AB = np.concatenate([im_A, im_B], 1)
            cv2.imwrite(path_AB, im_AB)

if __name__ == '__main__':
    main()
