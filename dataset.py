import os

import numpy
from PIL import Image
import six

from chainer.dataset import dataset_mixin

def _resize(img, size, resample, dtype):
    return numpy.asarray(Image.fromarray(img).resize((size, size), resample=resample), dtype=dtype)

def _crop(img, sx, sy, crop_size):
    if img.ndim == 2:
        img = img[:, :, numpy.newaxis]
    return img[sx:sx+crop_size, sy:sy+crop_size, :]

def _read_image_as_array(path, dtype, load_size, crop_size, flip):
    f = Image.open(path)

    A, B = numpy.array_split(numpy.asarray(f), 2, axis=1)
    A = _resize(A, load_size, Image.BILINEAR, dtype)
    B = _resize(B, load_size, Image.NEAREST, dtype)

    sx, sy = numpy.random.randint(0, load_size-crop_size, 2)
    A = _crop(A, sx, sy, crop_size)
    B = _crop(B, sx, sy, crop_size)
    
    if flip and numpy.random.rand() > 0.5:
        A = numpy.fliplr(A)
        B = numpy.fliplr(B)

    return A.transpose(2, 0, 1), B.transpose(2, 0, 1)

class Pix2pixImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, paths, load_size, crop_size, flip, AtoB, root='.', dtype=numpy.float32):
        if isinstance(paths, six.string_types):
            with open(paths) as paths_file:
                paths = [path.strip() for path in paths_file]
        self._paths = paths
        self._root = root
        self._dtype = dtype
        self.load_size = load_size
        self.crop_size = crop_size
        self.flip = flip
        self.AtoB = AtoB

    def __len__(self):
        return len(self._paths)

    def get_name(self, i):
        return os.path.basename(self._paths[i])

    def get_example(self, i):
        path = os.path.join(self._root, self._paths[i])
        A, B = _read_image_as_array(path, self._dtype, self.load_size, self.crop_size, self.flip)

        return (A, B) if self.AtoB else (B, A)