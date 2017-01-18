from __future__ import print_function
import argparse
import os

import chainer
from chainer import training, serializers
from chainer.training import extensions

from model import Generator, Discriminator
from updater import Pix2pixUpdater
from dataset import Pix2pixImageDataset
from utils import display_image

def make_optimizer(model, alpha, beta1):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
    optimizer.setup(model)
    return optimizer

def main():
    parser = argparse.ArgumentParser(description='pix2pix --- GAN for Image to Image translation')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', type=int, default=200, help='# of sweeps over training dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each mini-batch')
    parser.add_argument('--load_size', type=int, default=286, help='Scale image to load_size')
    parser.add_argument('--crop_size', type=int, default=256, help='After scale, crop image to crop_size')
    parser.add_argument('--flip', type=bool, default=True, help='If flip the images for data argumentation')
    parser.add_argument('--g_filter_num', type=int, default=64, help="# of filters in G's 1st conv layer")
    parser.add_argument('--d_filter_num', type=int, default=64, help="# of filters in D's 1st conv layer")
    parser.add_argument('--output_channel', type=int, default=3, help='# of output image channels')
    parser.add_argument('--n_layers', type=int, default=3, help='# of hidden layers in D')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for Adam')
    parser.add_argument('--beta1', type=float, default=0.5, help='Momentum term of adam')
    parser.add_argument('--lambd', type=float, default=100, help='Weight on L1 term in objective')
    parser.add_argument('--data_root', default='datasets', help='Folder containing train, val & test subfolder, as well as train.txt, val.txt, test.txt')
    parser.add_argument('--resume', default='', help='Resume the training from snapshot')
    parser.add_argument('--AtoB', type=bool, default=False, help='BtoA if False')
    parser.add_argument('--out', default='result', help='Directory to output the result')
    parser.add_argument('--snapshot_interval', type=int, default=10, help='Interval of snapshot (epoch)')
    parser.add_argument('--print_interval', type=int, default=1, help='Interval of printing log to console')
    parser.add_argument('--plot_interval', type=int, default=100, help='Interval of plot')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batch_size))
    print('# epoch: {}'.format(args.epoch))
    print('')
    print('# lr: {}'.format(args.lr))
    print('# beta1: {}'.format(args.beta1))
    print('# lambda: {}'.format(args.lambd))
    print('')
    print(args)
    print('')

    # Set up GAN G-D
    G = Generator(args.g_filter_num, args.output_channel)
    D = Discriminator(args.n_layers, args.d_filter_num)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        G.to_gpu()                               # Copy the model to the GPU
        D.to_gpu()

    # Setup an optimizer
    opt_G = make_optimizer(G, args.lr, args.beta1)
    opt_D = make_optimizer(D, args.lr, args.beta1)

    # Setup dataset & iterator
    trainset = Pix2pixImageDataset(os.path.join(args.data_root, 'train.txt'), args.load_size, args.crop_size, args.flip, args.AtoB, root=os.path.join(args.data_root, 'train'))
    valset = Pix2pixImageDataset(os.path.join(args.data_root, 'val.txt'), args.load_size, args.crop_size, args.flip, args.AtoB, root=os.path.join(args.data_root, 'val'))

    print('Trainset contains {} image files'.format(len(trainset)))
    print('Valset contains {} image files'.format(len(valset)))
    print('')

    train_iter = chainer.iterators.MultiprocessIterator(trainset, args.batch_size)

    # Set up a trainer
    updater = Pix2pixUpdater(
        models=(G, D),
        lambd=args.lambd,
        iterator=train_iter,
        optimizer={'main': opt_G, 'D': opt_D},
        device=args.gpu
    )
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Snapshot
    snapshot_interval = (args.snapshot_interval, 'epoch')
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        G, 'gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        D, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)

    # Display
    print_interval = (args.print_interval, 'iteration')
    trainer.extend(extensions.LogReport(trigger=print_interval))
    trainer.extend(extensions.PrintReport([
        'iteration', 'main/loss', 'main/loss_D', 'main/loss_l1', 'D/loss'
    ]), trigger=print_interval)
    trainer.extend(extensions.ProgressBar(update_interval=args.print_interval))

    trainer.extend(extensions.dump_graph('main/loss_l1', out_name='G_loss_L1.dot'))
    trainer.extend(extensions.dump_graph('main/loss', out_name='G_loss.dot'))
    trainer.extend(extensions.dump_graph('main/loss_D', out_name='G_loss_GAN.dot'))
    trainer.extend(extensions.dump_graph('D/loss', out_name='D_loss_GAN.dot'))

    # Plot
    plot_interval = (args.plot_interval, 'iteration')

    trainer.extend(
        extensions.PlotReport(['main/loss_D'], 'iteration', file_name='G_GAN_loss.png', trigger=plot_interval), trigger=plot_interval)
    trainer.extend(
        extensions.PlotReport(['D/loss'], 'iteration', file_name='D_GAN_loss.png', trigger=plot_interval), trigger=plot_interval)
    trainer.extend(
        extensions.PlotReport(['main/loss_l1'], 'iteration', file_name='G_L1_loss.png', trigger=plot_interval), trigger=plot_interval)
    trainer.extend(
        extensions.PlotReport(['main/loss'], 'iteration', file_name='G_loss.png', trigger=plot_interval), trigger=plot_interval)
    trainer.extend(display_image(G, valset, args.out, args.gpu), trigger=plot_interval)

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

if __name__ == '__main__':
    main()