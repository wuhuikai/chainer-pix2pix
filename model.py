import chainer
import chainer.links as L
import chainer.functions as F

class Generator(chainer.Chain):
    """
        U-net
        Input: nc x 256 x 256
    """
    def __init__(self, feature_map_nc, output_nc, w_init=None):
        super(Generator, self).__init__(
            c1=L.Convolution2D(None, feature_map_nc, ksize=4, stride=2, pad=1, initialW=w_init),
            c2=L.Convolution2D(None, 2*feature_map_nc, ksize=4, stride=2, pad=1, initialW=w_init),
            c3=L.Convolution2D(None, 4*feature_map_nc, ksize=4, stride=2, pad=1, initialW=w_init),
            c4=L.Convolution2D(None, 8*feature_map_nc, ksize=4, stride=2, pad=1, initialW=w_init),
            c5=L.Convolution2D(None, 8*feature_map_nc, ksize=4, stride=2, pad=1, initialW=w_init),
            c6=L.Convolution2D(None, 8*feature_map_nc, ksize=4, stride=2, pad=1, initialW=w_init),
            c7=L.Convolution2D(None, 8*feature_map_nc, ksize=4, stride=2, pad=1, initialW=w_init),
            c8=L.Convolution2D(None, 8*feature_map_nc, ksize=4, stride=2, pad=1, initialW=w_init),
            dc1=L.Deconvolution2D(None, 8*feature_map_nc, ksize=4, stride=2, pad=1, initialW=w_init),
            dc2=L.Deconvolution2D(None, 8*feature_map_nc, ksize=4, stride=2, pad=1, initialW=w_init),
            dc3=L.Deconvolution2D(None, 8*feature_map_nc, ksize=4, stride=2, pad=1, initialW=w_init),
            dc4=L.Deconvolution2D(None, 8*feature_map_nc, ksize=4, stride=2, pad=1, initialW=w_init),
            dc5=L.Deconvolution2D(None, 4*feature_map_nc, ksize=4, stride=2, pad=1, initialW=w_init),
            dc6=L.Deconvolution2D(None, 2*feature_map_nc, ksize=4, stride=2, pad=1, initialW=w_init),
            dc7=L.Deconvolution2D(None, feature_map_nc, ksize=4, stride=2, pad=1, initialW=w_init),
            dc8=L.Deconvolution2D(None, output_nc, ksize=4, stride=2, pad=1, initialW=w_init),
            b2=L.BatchNormalization(2*feature_map_nc),
            b3=L.BatchNormalization(4*feature_map_nc),
            b4=L.BatchNormalization(8*feature_map_nc),
            b5=L.BatchNormalization(8*feature_map_nc),
            b6=L.BatchNormalization(8*feature_map_nc),
            b7=L.BatchNormalization(8*feature_map_nc),
            b8=L.BatchNormalization(8*feature_map_nc),
            b1_d=L.BatchNormalization(8*feature_map_nc),
            b2_d=L.BatchNormalization(8*feature_map_nc),
            b3_d=L.BatchNormalization(8*feature_map_nc),
            b4_d=L.BatchNormalization(8*feature_map_nc),
            b5_d=L.BatchNormalization(4*feature_map_nc),
            b6_d=L.BatchNormalization(2*feature_map_nc),
            b7_d=L.BatchNormalization(feature_map_nc)
        )

    def __call__(self, x, test=False, dropout=True):
        e1 = self.c1(x)
        e2 = self.b2(self.c2(F.leaky_relu(e1)), test=test)
        e3 = self.b3(self.c3(F.leaky_relu(e2)), test=test)
        e4 = self.b4(self.c4(F.leaky_relu(e3)), test=test)
        e5 = self.b5(self.c5(F.leaky_relu(e4)), test=test)
        e6 = self.b6(self.c6(F.leaky_relu(e5)), test=test)
        e7 = self.b7(self.c7(F.leaky_relu(e6)), test=test)
        e8 = self.b8(self.c8(F.leaky_relu(e7)), test=test)
        d1 = F.concat((F.dropout(self.b1_d(self.dc1(F.relu(e8)), test=test), train=dropout), e7))
        d2 = F.concat((F.dropout(self.b2_d(self.dc2(F.relu(d1)), test=test), train=dropout), e6))
        d3 = F.concat((F.dropout(self.b3_d(self.dc3(F.relu(d2)), test=test), train=dropout), e5))
        d4 = F.concat((self.b4_d(self.dc4(F.relu(d3)), test=test), e4))
        d5 = F.concat((self.b5_d(self.dc5(F.relu(d4)), test=test), e3))
        d6 = F.concat((self.b6_d(self.dc6(F.relu(d5)), test=test), e2))
        d7 = F.concat((self.b7_d(self.dc7(F.relu(d6)), test=test), e1))
        y = F.tanh(self.dc8(F.relu(d7)))
        
        return y

class Discriminator(chainer.Chain):
    """
        PatchGAN
        Input: nc x 256 x 256
    """
    def __init__(self, n_layers, feature_map_nc, w_init=None):
        self.n_layers = n_layers
        layers = {
            'c0': L.Convolution2D(None, feature_map_nc, ksize=4, stride=2, pad=1, initialW=w_init)
        }
        for idx in range(1, n_layers):
            nc_mult = min(2**idx, 8)
            layers['c{}'.format(idx)] = L.Convolution2D(None, feature_map_nc*nc_mult, ksize=4, stride=2, pad=1, initialW=w_init)
            layers['b{}'.format(idx)] = L.BatchNormalization(feature_map_nc*nc_mult)
        nc_mult = min(2**n_layers, 8)
        layers['c{}'.format(n_layers)] = L.Convolution2D(None, feature_map_nc*nc_mult, ksize=4, stride=1, pad=1, initialW=w_init)
        layers['b{}'.format(n_layers)] = L.BatchNormalization(feature_map_nc*nc_mult)
        layers['c'] = L.Convolution2D(None, 1, ksize=4, stride=1, pad=1, initialW=w_init)

        super(Discriminator, self).__init__(**layers)

    def __call__(self, x, test=False):
        h = F.leaky_relu(self.c0(x))

        for idx in range(1, self.n_layers):
            h = F.leaky_relu(self['b{}'.format(idx)](self['c{}'.format(idx)](h), test=test))

        h = F.leaky_relu(self['b{}'.format(self.n_layers)](self['c{}'.format(self.n_layers)](h), test=test))
        h = F.sigmoid(self.c(h))

        return h