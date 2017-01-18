import os

import numpy as np
from scipy.ndimage import zoom
from PIL import Image

import chainer
from chainer import Variable

def data_process(batch, converter=chainer.dataset.concat_examples, device=None, volatile='off'):
    return Variable(converter(batch, device) / 127.5 - 1.0, volatile=volatile)

def output2img(y):
    y = chainer.cuda.to_cpu(y.data)
    return np.asarray((np.transpose(y, (0, 2, 3, 1)) + 1.0) * 127.5, dtype=np.uint8)

def display_image(G, valset, dst, device):
    @chainer.training.make_extension()
    def make_image(trainer):
        preview_dir = '{}/preview'.format(dst)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)

        idx = np.random.randint(0, len(valset))

        A, _ = valset.get_example(idx)
        name = valset.get_name(idx)
        fake_B = G(data_process([A], device=device, volatile='on'), test=True)
        fake_B = np.squeeze(output2img(fake_B))

        name = os.path.splitext(name)[0]
        preview_path = preview_dir + '/{}_iter_{}.png'.format(name, trainer.updater.iteration)
        Image.fromarray(fake_B).save(preview_path)

    return make_image