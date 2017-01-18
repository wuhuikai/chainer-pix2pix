import argparse

import os
import numpy as np
import glob

def write2txt(img_list, list_dir, file_name):
    with open(os.path.join(list_dir, file_name), 'w') as f:
        f.write('\n'.join(img_list))

def main():
    parser = argparse.ArgumentParser(description='Split cityscapes into train, val & test set')
    parser.add_argument('--root', type=str, default='/data1/wuhuikai/benchmark/cityscapes/leftImg8bit', help='Root folder for images of cityscapes')
    parser.add_argument('--list', type=str, default='list', help='Folder for storeing image list')
    parser.add_argument('--ratio', type=float, default=0.05, help='Test or Val set ratio')
    parser.add_argument('--include_val', type=bool, default=True, help='Include val set of cityscapes ?')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    args = parser.parse_args()

    print('Image folder: {}'.format(args.root))
    print('List folder: {}'.format(args.list))
    print('Train:Val:Test = {}:{}:{}'.format(1-2*args.ratio, args.ratio, args.ratio))
    print('')

    splits = ['train', 'val'] if args.include_val else ['train']
    city_list = [os.path.join(args.root, sp, city) for sp in splits for city in os.listdir(os.path.join(args.root, sp))]
    
    image_list = []
    for city in city_list:
        image_list += glob.glob(os.path.join(city, '*.png'))
    
    image_list = ['{} {}'.format(img, img.replace('leftImg8bit', 'gtFine').replace('.png', '_color.png')) for img in image_list]
    
    np.random.seed(args.seed)
    np.random.shuffle(image_list)

    total_num = len(image_list)
    val_test_num = int(total_num*args.ratio)
    print('Total image num: {}'.format(total_num))
    print('Train image num: {}'.format(total_num - 2*val_test_num))
    print('Val/Test image num: {}'.format(val_test_num))
    
    train_set = image_list[2*val_test_num:]
    val_set = image_list[val_test_num:2*val_test_num]
    test_set = image_list[:val_test_num]

    if not os.path.isdir(args.list):
        os.makedirs(args.list)
    write2txt(train_set, args.list, 'train_list.txt')
    write2txt(val_set, args.list, 'val_list.txt')
    write2txt(test_set, args.list, 'test_list.txt')

if __name__ == '__main__':
    main()
