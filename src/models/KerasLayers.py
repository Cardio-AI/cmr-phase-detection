import tensorflow
import tensorflow as tf

from tensorflow import keras
import keras.layers as kl
from keras import backend as K
from keras.layers import Dropout, BatchNormalization, Activation
from keras.layers import Layer
from keras.layers import UpSampling2D as UpSampling2DInterpol
from keras.layers import UpSampling3D
from tensorflow.python.keras.utils import conv_utils
import numpy as np
import sys

__all__ = ['UpSampling2DInterpol', 'UpSampling3DInterpol', 'Euler2Matrix', 'ScaleLayer',
           'UnetWrapper', 'ConvEncoder', 'conv_layer_fn', 'downsampling_block_fn',
           'upsampling_block_fn','Inverse3DMatrix', 'ConvDecoder', 'ConvEncoder', 'get_centers_tf',
           'get_idxs_tf', 'get_angle_tf', 'ComposeTransform', 'ConvBlock']


class UpSampling3DInterpol(UpSampling3D):

    def __init__(self, size=(1, 2, 2), interpolation='bilinear', **kwargs):
        self.size = conv_utils.normalize_tuple(size, 3, 'size')
        self.x = int(size[1])
        self.y = int(size[2])
        self.interpolation = interpolation
        super(self.__class__, self).__init__(**kwargs)

    def call(self, x):
        """
        :param x:
        :return:
        """
        target_size = (x.shape[2] * self.x, x.shape[3] * self.y)
        # traverse along the 3D volumes, handle the z-slices as batch
        return K.stack(
            tf.map_fn(lambda images:
                      tf.image.resize(
                          images=images,
                          size=target_size,
                          method=self.interpolation,  # define bilinear or nearest neighbor
                          preserve_aspect_ratio=True),
                      x))

    def get_config(self):
        config = super(UpSampling3DInterpol, self).get_config()
        config.update({'interpolation': self.interpolation, 'size': self.size})
        return config


class Inverse3DMatrix(Layer):
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)

    def call(self, m, **kwargs):
        """
        Calculate the inverse for an affine matrix
        :param m:
        :return:
        """
        # get the inverse of the affine matrix
        batch_size = tf.shape(m)[0]
        m_matrix = tf.keras.layers.Reshape(target_shape=(3, 4))(m)

        # Create a tensor with (b,1,4) and concat it to the affine matrix tensor (b,3,4)
        # and create a square tensor with (b,4,4)
        # (hack to slice the transformation matrix into an identity matrix)
        # don't know how to assign values to a tensor such as in numpy
        one = tf.ones((batch_size, 1, 1), dtype=tf.float16)
        zero = tf.zeros((batch_size, 1, 1), dtype=tf.float16)
        row = tf.concat([zero, zero, zero, one], axis=-1)
        ident = tf.concat([m_matrix, row], axis=1)

        m_matrix_inv = tf.linalg.inv(ident)

        m_inv = m_matrix_inv[:, :3, :]  # cut off the last row
        return tf.keras.layers.Flatten()(m_inv)

    def get_config(self):
        """ __init__() is overwritten, need to override this method to enable model.to_json() for this layer"""
        config = super(self.__class__, self).get_config()
        return config


class Euler2Matrix(Layer):
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)

    def call(self, theta, **kwargs):
        euler_1 = tf.expand_dims(theta[:, 0], -1)
        euler_2 = tf.expand_dims(theta[:, 1], -1)
        euler_3 = tf.expand_dims(theta[:, 2], -1)

        # clip values in a range -pi to pi, transformation is only defined within this range
        # clipping so far not necessary and not tested
        # pi = tf.constant(m.pi)
        # euler_1 = tf.clip_by_value(euler_1, -pi, pi)
        # euler_2 = tf.clip_by_value(euler_2, -pi, pi)
        # euler_3 = tf.clip_by_value(euler_3, -pi, pi)

        one = tf.ones_like(euler_1)
        zero = tf.zeros_like(euler_1)

        rot_x = tf.stack([tf.concat([one, zero, zero], axis=1),
                          tf.concat([zero, tf.cos(euler_1), tf.sin(euler_1)], axis=1),
                          tf.concat([zero, -tf.sin(euler_1), tf.cos(euler_1)], axis=1)], axis=1)

        rot_y = tf.stack([tf.concat([tf.cos(euler_2), zero, -tf.sin(euler_2)], axis=1),
                          tf.concat([zero, one, zero], axis=1),
                          tf.concat([tf.sin(euler_2), zero, tf.cos(euler_2)], axis=1)], axis=1)

        rot_z = tf.stack([tf.concat([tf.cos(euler_3), tf.sin(euler_3), zero], axis=1),
                          tf.concat([-tf.sin(euler_3), tf.cos(euler_3), zero], axis=1),
                          tf.concat([zero, zero, one], axis=1)], axis=1)

        rot_matrix = tf.matmul(rot_z, tf.matmul(rot_y, rot_x))

        # Extend matrix by the translation parameters
        translation = tf.expand_dims(tf.stack([theta[:, 3], theta[:, 4], theta[:, 5]], axis=-1), axis=-1)
        rot_matrix = tf.concat([rot_matrix, translation], axis=2)
        rot_matrix = tf.keras.layers.Flatten()(rot_matrix)
        return rot_matrix

    def get_config(self):
        """ __init__() is overwritten, need to override this method to enable model.to_json() for this layer"""
        config = super(self.__class__, self).get_config()
        return config


class ScaleLayer(Layer):
    def __init__(self, units=1., **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.scale = tf.Variable(units)

    def call(self, inputs, **kwargs):
        return inputs * self.scale

    def get_config(self):
        """ __init__() is overwritten, need to override this method to enable model.to_json() for this layer"""
        config = super(self.__class__, self).get_config()
        return config


class UnetWrapper(Layer):
    def __init__(self, unet, unet_inplane=(224, 224), resize=True, trainable=False, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.unet = unet
        self.unet.trainable = trainable
        self.unet_inplane = unet_inplane

        self.resize = resize

    def call(self, x, **kwargs):
        """
        3D wrapper for a 2D U-Net, any inplane shape is possible, layer will resize x to the U-Net input shape
        Parameters
        ----------
        x :
        kwargs :

        Returns
        -------

        """

        x = tf.unstack(x, axis=1)
        input_size = x[0].shape[1:-1]
        if self.resize:
            x = [self.unet(
                tf.compat.v1.image.resize(
                    images,
                    size=self.unet_inplane,
                    method=tf.image.ResizeMethod.BILINEAR,
                    align_corners=True,
                    name='resize')) for images in x]
            x = [tf.compat.v1.image.resize(img,
                                           size=input_size,
                                           method=tf.image.ResizeMethod.BILINEAR,
                                           align_corners=True,
                                           name='reverse-resize') for img in x]
        else:
            x = [self.unet(img) for img in x]

        x = tf.stack(x, axis=1)
        return x

    def get_config(self):
        """ __init__() is overwritten, need to override this method to enable model.to_json() for this layer"""
        config = super(self.__class__, self).get_config()
        config.update(self.unet.get_config())
        config.update({'unet_inplane': self.unet_inplane})
        return config


class ConvEncoder(Layer):
    def __init__(self, activation, batch_norm, bn_first, depth, drop_3, dropouts, f_size, filters,
                 kernel_init, m_pool, ndims, pad):
        """
        Convolutional encoder for 2D or 3D input images/volumes.
        The architecture is aligned to the downsampling part of a U-Net
        Parameters
        ----------
        activation :
        batch_norm :
        bn_first :
        depth :
        drop_3 :
        dropouts :
        f_size :
        filters :
        kernel_init :
        m_pool :
        ndims :
        pad :
        """
        super(self.__class__, self).__init__()
        self.activation = activation
        self.batch_norm = batch_norm
        self.bn_first = bn_first
        self.depth = depth
        self.drop_3 = drop_3
        self.dropouts = dropouts
        self.f_size = f_size
        self.filters = filters
        self.kernel_init = kernel_init
        self.m_pool = m_pool
        self.ndims = ndims
        self.pad = pad

        self.downsamplings = []
        filters = self.filters

        for l in range(self.depth):
            db = DownSampleBlock(filters=filters,
                                 f_size=self.f_size,
                                 activation=self.activation,
                                 drop=self.dropouts[l],
                                 batch_norm=self.batch_norm,
                                 kernel_init=self.kernel_init,
                                 pad=self.pad,
                                 m_pool=self.m_pool,
                                 bn_first=self.bn_first,
                                 ndims=self.ndims)
            self.downsamplings.append(db)
            filters *= 2

        self.conv1 = ConvBlock(filters=filters, f_size=self.f_size,
                               activation=self.activation, batch_norm=self.batch_norm, kernel_init=self.kernel_init,
                               pad=self.pad, bn_first=self.bn_first, ndims=self.ndims)

        self.bn = Dropout(self.drop_3)
        self.conv2 = ConvBlock(filters=filters, f_size=self.f_size,
                               activation=self.activation, batch_norm=self.batch_norm, kernel_init=self.kernel_init,
                               pad=self.pad, bn_first=self.bn_first, ndims=self.ndims)

    def call(self, inputs, **kwargs):

        encs = []
        skips = []

        self.first_block = True

        for db in self.downsamplings:

            if self.first_block:
                # first block
                input_tensor = inputs
                self.first_block = False
            else:
                # all other blocks, use the max-pooled output of the previous encoder block as input
                # remember the max-pooled output from the previous layer
                input_tensor = encs[-1]

            skip, enc = db(input_tensor)
            encs.append(enc)
            skips.append(skip)

        # return the last encoding block result
        encoding = encs[-1]
        encoding = self.conv1(inputs=encoding)
        encoding = self.bn(encoding)
        encoding = self.conv2(inputs=encoding)

        # work as u-net encoder or cnn encoder
        return encoding, skips

    def get_config(self):
        """ __init__() is overwritten, need to override this method to enable model.to_json() for this layer"""

        config = super(self.__class__, self).get_config()
        config.update({'activation': self.activation,
                       'batch_norm': self.batch_norm,
                       'bn_first': self.bn_first,
                       'depth': self.depth,
                       'drop_3': self.drop_3,
                       'dropouts': self.dropouts,
                       'f_size': self.f_size,
                       'filters': self.filters,
                       'kernel_init': self.kernel_init,
                       'm_pool': self.m_pool,
                       'ndims': self.ndims,
                       'pad': self.pad})
        return config


class ConvDecoder(Layer):
    def __init__(self, activation, batch_norm, bn_first, depth, drop_3, dropouts, f_size, filters,
                 kernel_init, up_size, ndims, pad):
        """
        Convolutional Decoder path, could be used in combination with the encoder layer,
        or as up-scaling path for super resolution etc.
        Parameters
        ----------
        activation :
        batch_norm :
        bn_first :
        depth :
        drop_3 :
        dropouts :
        f_size :
        filters :
        kernel_init :
        up_size :
        ndims :
        pad :
        """
        super(self.__class__, self).__init__()
        self.activation = activation
        self.batch_norm = batch_norm
        self.bn_first = bn_first
        self.depth = depth
        self.drop_3 = drop_3
        self.dropouts = dropouts
        self.f_size = f_size
        self.filters = filters
        self.kernel_init = kernel_init
        self.up_size = up_size
        self.ndims = ndims
        self.pad = pad
        self.upsamplings = []

        filters = self.filters
        for layer in range(self.depth):
            ub = UpSampleBlock(filters=filters,
                               f_size=self.f_size,
                               activation=self.activation,
                               drop=self.dropouts[layer],
                               batch_norm=self.batch_norm,
                               kernel_init=self.kernel_init,
                               pad=self.pad,
                               up_size=self.up_size,
                               bn_first=self.bn_first,
                               ndims=self.ndims)
            self.upsamplings.append(ub)
            filters /= 2

    def call(self, inputs, **kwargs):

        enc, skips = inputs

        for upsampling in self.upsamplings:
            skip = skips.pop()
            enc = upsampling([enc, skip])

        return enc

    def get_config(self):
        """ __init__() is overwritten, need to override this method to enable model.to_json() for this layer"""

        config = super(self.__class__, self).get_config()
        config.update({'activation': self.activation,
                       'batch_norm': self.batch_norm,
                       'bn_first': self.bn_first,
                       'depth': self.depth,
                       'drop_3': self.drop_3,
                       'dropouts': self.dropouts,
                       'f_size': self.f_size,
                       'filters': self.filters,
                       'kernel_init': self.kernel_init,
                       'up_size': self.up_size,
                       'ndims': self.ndims,
                       'pad': self.pad})
        return config


class ConvBlock(Layer):
    def __init__(self, filters=16, f_size=(3, 3, 3), activation='elu', batch_norm=True, kernel_init='he_normal',
                 pad='same', bn_first=False, ndims=2, strides=1):
        """
        Wrapper for a 2/3D-conv layer + batchnormalisation
        Either with Conv,BN,activation or Conv,activation,BN

        :param filters: int, number of filters
        :param f_size: tuple of int, filterzise per axis
        :param activation: string, which activation function should be used
        :param batch_norm: bool, use batch norm or not
        :param kernel_init: string, keras enums for kernel initialisation
        :param pad: string, keras enum how to pad, the conv
        :param bn_first: bool, decide if we want to apply the BN before the conv. operation or afterwards
        :param ndims: int, define the conv dimension
        :param strides: int, stride of the conv filter
        :return: a functional tf.keras conv block
        expects an numpy or tensor object with (batchsize,z,x,y,channels)
        """
        super(self.__class__, self).__init__()
        self.activation = activation
        self.batch_norm = batch_norm
        self.bn_first = bn_first
        self.f_size = f_size
        self.filters = filters
        self.kernel_init = kernel_init
        self.ndims = ndims
        self.pad = pad
        self.strides = strides
        self.encoder = list()

        # create the layers
        Conv = getattr(kl, 'Conv{}D'.format(self.ndims))
        f_size = self.f_size[:self.ndims]

        self.conv = Conv(filters=self.filters, kernel_size=f_size, kernel_initializer=self.kernel_init,
                         padding=self.pad, strides=self.strides)
        self.conv_activation = Conv(filters=self.filters, kernel_size=f_size, kernel_initializer=self.kernel_init,
                                    padding=self.pad, strides=self.strides, activation=activation)
        self.bn = BatchNormalization(axis=-1)
        self.activation = Activation(self.activation)

    def call(self, inputs, **kwargs):

        if self.bn_first:
            # , kernel_regularizer=regularizers.l2(0.0001)
            conv1 = self.conv(inputs)
            conv1 = self.bn(conv1) if self.batch_norm else conv1
            conv1 = self.activation(conv1)

        else:
            # kernel_regularizer=regularizers.l2(0.0001),
            conv1 = self.conv_activation(inputs)
            conv1 = self.bn(conv1) if self.batch_norm else conv1

        return conv1

    def get_config(self):
        config = super(self.__class__, self).get_config()
        config.update({'activation': self.activation,
                       'batch_norm': self.batch_norm,
                       'bn_first': self.bn_first,
                       'f_size': self.f_size,
                       'filters': self.filters,
                       'kernel_init': self.kernel_init,
                       'ndims': self.ndims,
                       'pad': self.pad})
        return config


class DownSampleBlock(Layer):
    def __init__(self, filters=16, f_size=(3, 3, 3), activation='elu', drop=0.3, batch_norm=True,
                 kernel_init='he_normal', pad='same', m_pool=(2, 2), bn_first=False, ndims=2):
        """
    Create an 2D/3D-downsampling block for the u-net architecture
    :param filters: int, number of filters per conv-layer
    :param f_size: tuple of int, filtersize per axis
    :param activation: string, which activation function should be used
    :param drop: float, define the dropout rate between the conv layers of this block
    :param batch_norm: bool, use batch norm or not
    :param kernel_init: string, keras enums for kernel initialisation
    :param pad: string, keras enum how to pad, the conv
    :param m_pool: tuple of int, size of the max-pooling filters
    :param bn_first: bool, decide if we want to apply the BN before the conv. operation or afterwards
    :param ndims: int, define the conv dimension
    :return: a functional tf.keras upsampling block
    Excpects a numpy or tensor input with (batchsize,z,x,y,channels)
    """

        super(self.__class__, self).__init__()
        self.activation = activation
        self.batch_norm = batch_norm
        self.bn_first = bn_first
        self.drop = drop
        self.f_size = f_size
        self.filters = filters
        self.kernel_init = kernel_init
        self.m_pool = m_pool
        self.ndims = ndims
        self.pad = pad
        self.encoder = list()

        self.m_pool = self.m_pool[-self.ndims:]
        self.pool_fn = getattr(kl, 'MaxPooling{}D'.format(self.ndims))
        self.pool = self.pool_fn(m_pool)
        self.conf1 = ConvBlock(filters=self.filters, f_size=self.f_size, activation=self.activation,
                               batch_norm=self.batch_norm,
                               kernel_init=self.kernel_init, pad=self.pad, bn_first=self.bn_first, ndims=self.ndims)
        self.dropout = Dropout(self.drop)
        self.conf2 = ConvBlock(filters=self.filters, f_size=self.f_size, activation=self.activation,
                               batch_norm=self.batch_norm,
                               kernel_init=self.kernel_init, pad=self.pad, bn_first=self.bn_first, ndims=self.ndims)

    def call(self, x, **kwargs):
        x = self.conf1(x)
        x = self.dropout(x)
        conv1 = self.conf2(x)
        p1 = self.pool(conv1)

        return [conv1, p1]


class UpSampleBlock(Layer):
    def __init__(self, use_upsample=True, filters=16, f_size=(3, 3, 3), activation='elu',
                 drop=0.3, batch_norm=True, kernel_init='he_normal', pad='same', up_size=(2, 2), bn_first=False,
                 ndims=2):
        """
        Create an upsampling block for the u-net architecture
        Each blocks consists of these layers: upsampling/transpose,concat,conv,dropout,conv
        Either with "upsampling,conv" or "transpose" upsampling
        :param use_upsample: bool, whether to use upsampling or transpose layer
        :param filters: int, number of filters per conv-layer
        :param f_size: tuple of int, filter size per axis
        :param activation: string, which activation function should be used
        :param drop: float, define the dropout rate between the conv layers of this block
        :param batch_norm: bool, use batch norm or not
        :param kernel_init: string, keras enums for kernel initialisation
        :param pad: string, keras enum how to pad, the conv
        :param up_size: tuple of int, size of the upsampling filters, either by transpose layers or upsampling layers
        :param bn_first: bool, decide if we want to apply the BN before the conv. operation or afterwards
        :param ndims: int, define the conv dimension
        :return: a functional tf.keras upsampling block
        Expects an input with length 2 lower block: batchsize,z,x,y,channels, skip layers: batchsize,z,x,y,channels
        """

        super(self.__class__, self).__init__()
        self.activation = activation
        self.batch_norm = batch_norm
        self.bn_first = bn_first
        self.drop = drop
        self.f_size = f_size
        self.filters = filters
        self.kernel_init = kernel_init
        self.use_upsample = use_upsample
        self.up_size = up_size
        self.ndims = ndims
        self.pad = pad
        self.encoder = list()

        Conv = getattr(kl, 'Conv{}D'.format(self.ndims))
        UpSampling = getattr(kl, 'UpSampling{}D'.format(self.ndims))
        ConvTranspose = getattr(kl, 'Conv{}DTranspose'.format(self.ndims))

        f_size = self.f_size[-self.ndims:]

        # use upsample&conv or transpose layer
        self.upsample = UpSampling(size=self.up_size)
        self.conv1 = Conv(filters=self.filters, kernel_size=f_size, padding=self.pad,
                          kernel_initializer=self.kernel_init,
                          activation=self.activation)

        self.convTranspose = ConvTranspose(filters=self.filters, kernel_size=f_size, strides=self.up_size,
                                           padding=self.pad,
                                           kernel_initializer=self.kernel_init,
                                           activation=self.activation)

        self.concatenate = tf.keras.layers.Concatenate(axis=-1)

        self.convBlock1 = ConvBlock(filters=self.filters, f_size=f_size, activation=self.activation,
                                    batch_norm=self.batch_norm,
                                    kernel_init=self.kernel_init, pad=self.pad, bn_first=self.bn_first,
                                    ndims=self.ndims)
        self.dropout = Dropout(self.drop)
        self.convBlock2 = ConvBlock(filters=self.filters, f_size=f_size, activation=self.activation,
                                    batch_norm=self.batch_norm,
                                    kernel_init=self.kernel_init, pad=self.pad, bn_first=self.bn_first,
                                    ndims=self.ndims)

    def call(self, inputs, **kwargs):

        if len(inputs) == 2:
            skip = True
            lower_input, conv_input = inputs
        else:
            skip = False
            lower_input = inputs

        # use upsample&conv or transpose layer
        if self.use_upsample:

            deconv1 = self.upsample(lower_input)
            deconv1 = self.conv1(deconv1)

        else:
            deconv1 = self.convTranspose(lower_input)

        # if skip given, concat
        if skip:
            deconv1 = self.concatenate([deconv1, conv_input])
        deconv1 = self.convBlock1(inputs=deconv1)
        deconv1 = self.dropout(deconv1)
        deconv1 = self.convBlock2(inputs=deconv1)

        return deconv1

    def get_config(self):
        """ __init__() is overwritten, need to override this method to enable model.to_json() for this layer"""
        config = super(self.__class__, self).get_config()
        config.update({'activation': self.activation,
                       'batch_norm': self.batch_norm,
                       'bn_first': self.bn_first,
                       'drop': self.drop,
                       'f_size': self.f_size,
                       'filters': self.filters,
                       'kernel_init': self.kernel_init,
                       'm_pool': self.m_pool,
                       'ndims': self.ndims,
                       'pad': self.pad})
        return config


def conv_layer_fn(inputs, filters=16, f_size=(3, 3, 3), activation='elu', batch_norm=True, kernel_init='he_normal',
                  pad='same', bn_first=False, ndims=2, custom_name=''):
    """
    Wrapper for a 2/3D-conv layer + batchnormalisation
    Either with Conv,BN,activation or Conv,activation,BN

    :param inputs: numpy or tensor object batchsize,z,x,y,channels
    :param filters: int, number of filters
    :param f_size: tuple of int, filterzise per axis
    :param activation: string, which activation function should be used
    :param batch_norm: bool, use batch norm or not
    :param kernel_init: string, keras enums for kernel initialisation
    :param pad: string, keras enum how to pad, the conv
    :param bn_first: bool, decide if we want to apply the BN before the conv. operation or afterwards
    :param ndims: int, define the conv dimension
    :return: a functional tf.keras conv block
    """

    Conv = getattr(kl, 'Conv{}D'.format(ndims))
    f_size = f_size[:ndims]

    if bn_first:
        # , kernel_regularizer=regularizers.l2(0.0001)
        conv1 = Conv(filters=filters, kernel_size=f_size, kernel_initializer=kernel_init, padding=pad)(inputs)
        conv1 = BatchNormalization(axis=-1)(conv1) if batch_norm else conv1
        conv1 = Activation(activation)(conv1)

    else:
        # kernel_regularizer=regularizers.l2(0.0001),
        conv1 = Conv(filters=filters, kernel_size=f_size, activation=activation, kernel_initializer=kernel_init,
                     padding=pad)(inputs)
        conv1 = BatchNormalization(axis=-1)(conv1) if batch_norm else conv1

    return conv1


def downsampling_block_fn(inputs, filters=16, f_size=(3, 3, 3), activation='elu', drop=0.3, batch_norm=True,
                          kernel_init='he_normal', pad='same', m_pool=(2, 2), bn_first=False, ndims=2):
    """
    Create an 2D/3D-downsampling block for the u-net architecture
    :param inputs: numpy or tensor input with batchsize,z,x,y,channels
    :param filters: int, number of filters per conv-layer
    :param f_size: tuple of int, filtersize per axis
    :param activation: string, which activation function should be used
    :param drop: float, define the dropout rate between the conv layers of this block
    :param batch_norm: bool, use batch norm or not
    :param kernel_init: string, keras enums for kernel initialisation
    :param pad: string, keras enum how to pad, the conv
    :param m_pool: tuple of int, size of the max-pooling layer
    :param bn_first: bool, decide if we want to apply the BN before the conv. operation or afterwards
    :param ndims: int, define the conv dimension
    :return: a functional tf.keras downsampling block
    """
    m_pool = m_pool[-ndims:]
    pool = getattr(kl, 'MaxPooling{}D'.format(ndims))

    conv1 = conv_layer_fn(inputs=inputs, filters=filters, f_size=f_size, activation=activation, batch_norm=batch_norm,
                          kernel_init=kernel_init, pad=pad, bn_first=bn_first, ndims=ndims)
    conv1 = Dropout(drop)(conv1)
    conv1 = conv_layer_fn(inputs=conv1, filters=filters, f_size=f_size, activation=activation, batch_norm=batch_norm,
                          kernel_init=kernel_init, pad=pad, bn_first=bn_first, ndims=ndims)
    p1 = pool(m_pool)(conv1)

    return (conv1, p1)


def upsampling_block_fn(lower_input, conv_input, use_upsample=True, filters=16, f_size=(3, 3, 3), activation='elu',
                        drop=0.3, batch_norm=True, kernel_init='he_normal', pad='same', up_size=(2, 2), bn_first=False,
                        ndims=2):
    """
    Create an upsampling block for the u-net architecture
    Each blocks consists of these layers: upsampling/transpose,concat,conv,dropout,conv
    Either with "upsampling,conv" or "transpose" upsampling
    :param lower_input: numpy input from the lower block: batchsize,z,x,y,channels
    :param conv_input: numpy input from the skip layers: batchsize,z,x,y,channels
    :param use_upsample: bool, whether to use upsampling or not
    :param filters: int, number of filters per conv-layer
    :param f_size: tuple of int, filtersize per axis
    :param activation: string, which activation function should be used
    :param drop: float, define the dropout rate between the conv layers of this block
    :param batch_norm: bool, use batch norm or not
    :param kernel_init: string, keras enums for kernel initialisation
    :param pad: string, keras enum how to pad, the conv
    :param up_size: tuple of int, size of the upsampling filters, either by transpose layers or upsampling layers
    :param bn_first: bool, decide if we want to apply the BN before the conv. operation or afterwards
    :param ndims: int, define the conv dimension
    :return: a functional tf.keras upsampling block
    """

    Conv = getattr(kl, 'Conv{}D'.format(ndims))
    f_size = f_size[-ndims:]

    # use upsample&conv or transpose layer
    if use_upsample:
        # import src.models.KerasLayers as ownkl
        # UpSampling = getattr(ownkl, 'UpSampling{}DInterpol'.format(ndims))
        UpSampling = getattr(kl, 'UpSampling{}D'.format(ndims))
        deconv1 = UpSampling(size=up_size)(lower_input)
        deconv1 = Conv(filters=filters, kernel_size=f_size, padding=pad, kernel_initializer=kernel_init,
                       activation=activation)(deconv1)

    else:
        ConvTranspose = getattr(kl, 'Conv{}DTranspose'.format(ndims))
        deconv1 = ConvTranspose(filters=filters, kernel_size=f_size, strides=up_size, padding=pad,
                                kernel_initializer=kernel_init,
                                activation=activation)(lower_input)

    deconv1 = tf.keras.layers.Concatenate(axis=-1)([deconv1, conv_input])

    deconv1 = conv_layer_fn(inputs=deconv1, filters=filters, f_size=f_size, activation=activation,
                            batch_norm=batch_norm,
                            kernel_init=kernel_init, pad=pad, bn_first=bn_first, ndims=ndims)
    deconv1 = Dropout(drop)(deconv1)
    deconv1 = conv_layer_fn(inputs=deconv1, filters=filters, f_size=f_size, activation=activation,
                            batch_norm=batch_norm,
                            kernel_init=kernel_init, pad=pad, bn_first=bn_first, ndims=ndims)

    return deconv1


# Downsampling part of a U-net, could be used to extract feature maps from a input 2D or 3D volume
def encoder_fn(activation, batch_norm, bn_first, depth, drop_3, dropouts, f_size, filters, inputs,
               kernel_init, m_pool, ndims, pad):
    """
    Encoder for 2d or 3d data, could be used for a U-net.
    Implementation based on the functional tf.keras api
    :param activation:
    :param batch_norm:
    :param bn_first:
    :param depth:
    :param drop_3:
    :param dropouts:
    :param f_size:
    :param filters:
    :param inputs:
    :param kernel_init:
    :param m_pool:
    :param ndims:
    :param pad:
    :return:
    """

    encoder = list()
    dropouts = dropouts.copy()

    # build the encoder
    for l in range(depth):

        if len(encoder) == 0:
            # first block
            input_tensor = inputs
        else:
            # all other blocks, use the max-pooled output of the previous encoder block
            # remember the max-pooled output from the previous layer
            input_tensor = encoder[-1][1]
        encoder.append(
            downsampling_block_fn(inputs=input_tensor,
                                  filters=filters,
                                  f_size=f_size,
                                  activation=activation,
                                  drop=dropouts[l],
                                  batch_norm=batch_norm,
                                  kernel_init=kernel_init,
                                  pad=pad,
                                  m_pool=m_pool,
                                  bn_first=bn_first,
                                  ndims=ndims))
        filters *= 2
    # middle part
    input_tensor = encoder[-1][1]
    fully = conv_layer_fn(inputs=input_tensor, filters=filters, f_size=f_size,
                          activation=activation, batch_norm=batch_norm, kernel_init=kernel_init,
                          pad=pad, bn_first=bn_first, ndims=ndims)
    fully = Dropout(drop_3)(fully)
    fully = conv_layer_fn(inputs=fully, filters=filters, f_size=f_size,
                          activation=activation, batch_norm=batch_norm, kernel_init=kernel_init,
                          pad=pad, bn_first=bn_first, ndims=ndims)
    return fully


def inverse_affine_matrix_fn(m):
    """
    Calculate the inverse for an affine matrix
    :param m:
    :return:
    """
    # get the inverse of the affine matrix
    batch_size = tf.shape(m)[0]
    m_matrix = tf.reshape(m, (batch_size, 3, 4))

    # concat a row with b,1,4 to b,3,4 and create a b,4,4
    # (hack to slice the transformation matrix into an identity matrix)
    # don't know how to assign values to a tensor such as in numpy
    one = tf.ones((batch_size, 1, 1))
    zero = tf.zeros((batch_size, 1, 1))
    row = tf.concat([zero, zero, zero, one], axis=-1)
    ident = tf.concat([m_matrix, row], axis=1)

    m_matrix_inv = tf.linalg.inv(ident)
    m_inv = m_matrix_inv[:, :3, :]  # cut off the last row
    return tf.keras.layers.Flatten()(m_inv)


# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix_fn(theta):
    """
    Calculate a rotation and translation matrix from 6 parameters given in theta
    The input is a list of tensors, the first three will be interpreted as euler angles
    The last three as translation params.
    :param theta: list of tensors with a length of 6 each with the shape b,1
    :return:
    """

    one = tf.ones_like(theta[0], dtype=tf.float32)
    zero = tf.zeros_like(theta[0], dtype=tf.float32)

    rot_x = tf.stack([tf.concat([one, zero, zero], axis=1),
                      tf.concat([zero, tf.cos(theta[0]), tf.sin(theta[0])], axis=1),
                      tf.concat([zero, -tf.sin(theta[0]), tf.cos(theta[0])], axis=1)], axis=1)

    rot_y = tf.stack([tf.concat([tf.cos(theta[1]), zero, -tf.sin(theta[1])], axis=1),
                      tf.concat([zero, one, zero], axis=1),
                      tf.concat([tf.sin(theta[1]), zero, tf.cos(theta[1])], axis=1)], axis=1)

    rot_z = tf.stack([tf.concat([tf.cos(theta[2]), tf.sin(theta[2]), zero], axis=1),
                      tf.concat([-tf.sin(theta[2]), tf.cos(theta[2]), zero], axis=1),
                      tf.concat([zero, zero, one], axis=1)], axis=1)

    rot_matrix = tf.matmul(rot_z, tf.matmul(rot_y, rot_x))

    # Apply learnable translation
    translation = tf.expand_dims(tf.stack([theta[3][:, 0], theta[4][:, 0], theta[5][:, 0]], axis=-1), axis=-1)
    # ignore translation
    # translation = tf.expand_dims(tf.stack([zero[:,0],zero[:, 0], zero[:, 0]], axis=-1), axis=-1)
    rot_matrix = tf.concat([rot_matrix, translation], axis=2)
    rot_matrix = tf.keras.layers.Flatten()(rot_matrix)

    # model = Model(inputs=[theta], outputs=rot_matrix, name='Affine_matrix_builder')

    return rot_matrix


def affineMatrixInverter_fn(m):
    """
    Concats an affine Matrix (b,12) to square shape, calculates the inverse and returns the sliced version
    :param m:
    :return: m inverted
    """
    # get the inverse of the affine matrix, rotate y back and compare it with AXtoSAXtoAX
    batch_size = tf.shape(m)[0]
    m_matrix = tf.reshape(m, (batch_size, 3, 4))
    # concat a row with b,1,4 to b,3,4 and create a b,4,4
    # (hack to slice the transformation matrix into an identity matrix)
    # don't know how to assign values to a tensor such as in numpy
    one = tf.ones((batch_size, 1, 1))
    zero = tf.zeros((batch_size, 1, 1))
    row = tf.concat([zero, zero, zero, one], axis=-1)
    ident = tf.concat([m_matrix, row], axis=1)

    m_matrix_inv = tf.linalg.inv(ident)
    m_inv = m_matrix_inv[:, :3, :]  # cut off the last row
    m_inv = tf.keras.layers.Flatten()(m_inv)

    return m_inv


flow2direction_lambda = tf.keras.layers.Lambda(
            lambda x: get_angle_tf(x[0], x[1]), name='flow2direction')
minmax_lambda = lambda x : x[1] + (((x[0] - np.min(x[0])) * (x[2]-x[1]))/(np.max(x[0]) - np.min(x[0]) + sys.float_info.epsilon))
def get_focus_tf(p,dim=[12,12,12]):
    return tf.cast(
        tf.tile(tf.convert_to_tensor([*p])[tf.newaxis, tf.newaxis, tf.newaxis, ...],
                (*dim, 1)), tf.float32)

# returns a matrix with the indicies as values, similar to np.indicies
def get_idxs_tf(x):
    return tf.cast(
        tf.reshape(tf.where(tf.ones((x[0], x[1], x[2]))), (x[0], x[1], x[2], 3)),
        tf.float32)


# returns a matrix with vectors pointing to the center
def get_centers_tf(x):
    return tf.cast(
        tf.tile(tf.convert_to_tensor([x[0] // 2, x[1] // 2, x[2] // 2])[tf.newaxis, tf.newaxis, tf.newaxis, ...],
                (x[0], x[1], x[2], 1)), tf.float32)

def get_angle_tf(a, b, indegree=False):
    """
    this should work for batches of n-dimensional vectors
    α = arccos[(a · b) / (|a| * |b|)]
    |v| = √(x² + y² + z²)
    in 3D space
    If vectors a = [xa, ya, za], b = [xb, yb, zb], then:
    α = arccos[(xa * xb + ya * yb + za * zb) / (√(xa2 + ya2 + za2) * √(xb2 + yb2 + zb2))]

    Args:
        a (tf.tensor): b,z,y,x,3
        b (tf.tensor): b,z,y,x,3
        indegree (bool): other-wise calc cos(angle)<- this is differentiable

    Returns: tf.tensor with the same shape except of the last axis

    """
    import math as m
    pi = tf.constant(m.pi)
    b = tf.cast(b, dtype=a.dtype)
    inner = tf.einsum('...i,...i->...', a, b)
    norms = tf.norm(a, ord='euclidean', axis=-1) * tf.norm(b, ord='euclidean', axis=-1)  # [...,None]
    cos = inner / (norms + sys.float_info.epsilon)
    if indegree:
        rad = tf.math.acos(tf.clip_by_value(cos, -1.0, 1.0)) # need to check if this is necessary
        # rad2deg conversion
        deg = rad * (180.0/pi)
        cos = deg
    return cos[..., tf.newaxis]

def get_angle_np(a, b, indegree=False):
    """
    this should work for batches of n-dimensional vectors
    This version is faster if we use it outside of the training
    e.g.: in a post-processing step of the deformable
    α = arccos[(a · b) / (|a| * |b|)]
    |v| = √(x² + y² + z²)
    in 3D space
    If vectors a = [xa, ya, za], b = [xb, yb, zb], then:
    α = arccos[(xa * xb + ya * yb + za * zb) / (√(xa2 + ya2 + za2) * √(xb2 + yb2 + zb2))]

    Args:
        a (np.ndarray): b,z,y,x,3
        b (np.ndarray): b,z,y,x,3
        indegree (bool): other-wise calc cos(angle)<- this is differentiable

    Returns: np.ndarray with the same shape except of the last axis

    """

    import math as m
    pi = m.pi
    #b = np.cast(b, dtype=a.dtype)
    inner = np.einsum('...i,...i->...', a, b)
    norms = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)  # [...,None]
    cos = inner / (norms + sys.float_info.epsilon)
    if indegree:
        rad = np.arccos(np.clip(cos, -1.0, 1.0))
        #rad = np.arccos(cos)
        # rad2deg conversion
        deg = rad * (180.0 / pi)
        cos = deg
    return cos[..., np.newaxis]

class SpatialTransformer(Layer):
    """
    N-D Spatial Transformer Tensorflow / Keras Layer
    The Layer can handle both affine and dense transforms.
    Both transforms are meant to give a 'shift' from the current position.
    Therefore, a dense transform gives displacements (not absolute locations) at each voxel,
    and an affine transform gives the *difference* of the affine matrix from
    the identity matrix (unless specified otherwise).
    If you find this function useful, please cite:
      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.
    Originally, this code was based on voxelmorph code, which
    was in turn transformed to be dense with the help of (affine) STN code
    via https://github.com/kevinzakka/spatial-transformer-network
    Since then, we've re-written the code to be generalized to any
    dimensions, and along the way wrote grid and interpolation functions
    """

    def __init__(self,
                 interp_method='linear',
                 indexing='ij',
                 single_transform=False,
                 fill_value=None,
                 add_identity=True,
                 shift_center=True,
                 **kwargs):
        """
        Parameters:
            interp_method: 'linear' or 'nearest'
            single_transform: whether a single transform supplied for the whole batch
            indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian)
                'xy' indexing will have the first two entries of the flow
                (along last axis) flipped compared to 'ij' indexing
            fill_value (default: None): value to use for points outside the domain.
                If None, the nearest neighbors will be used.
            add_identity (default: True): whether the identity matrix is added
                to affine transforms.
            shift_center (default: True): whether the grid is shifted to the center
                of the image when converting affine transforms to warp fields.
        """
        self.interp_method = interp_method
        self.fill_value = fill_value
        self.add_identity = add_identity
        self.shift_center = shift_center
        self.ndims = None
        self.inshape = None
        self.single_transform = single_transform

        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing

        super(self.__class__, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'interp_method': self.interp_method,
            'indexing': self.indexing,
            'single_transform': self.single_transform,
            'fill_value': self.fill_value,
            'add_identity': self.add_identity,
            'shift_center': self.shift_center,
        })
        return config

    def build(self, input_shape):
        """
        input_shape should be a list for two inputs:
        input1: image.
        input2: transform Tensor
            if affine:
                should be a N x N+1 matrix
                *or* a N*N+1 tensor (which will be reshape to N x (N+1) and an identity row added)
            if not affine:
                should be a *vol_shape x N
        """

        if len(input_shape) > 2:
            raise Exception('Spatial Transformer must be called on a list of length 2.'
                            'First argument is the image, second is the transform.')

        # set up number of dimensions
        self.ndims = len(input_shape[0]) - 2
        self.inshape = input_shape
        vol_shape = input_shape[0][1:-1]
        trf_shape = input_shape[1][1:]

        # the transform is an affine iff:
        # it's a 1D Tensor [dense transforms need to be at least ndims + 1]
        # it's a 2D Tensor and shape == [N+1, N+1] or [N, N+1]
        #   [dense with N=1, which is the only one that could have a transform shape of 2, would be of size Mx1]
        is_matrix = len(trf_shape) == 2 and trf_shape[0] in (self.ndims, self.ndims + 1) and trf_shape[
            1] == self.ndims + 1
        self.is_affine = len(trf_shape) == 1 or is_matrix

        # check sizes
        if self.is_affine and len(trf_shape) == 1:
            ex = self.ndims * (self.ndims + 1)
            if trf_shape[0] != ex:
                raise Exception('Expected flattened affine of len %d but got %d'
                                % (ex, trf_shape[0]))

        if not self.is_affine:
            if trf_shape[-1] != self.ndims:
                raise Exception('Offset flow field size expected: %d, found: %d'
                                % (self.ndims, trf_shape[-1]))

        # confirm built
        self.built = True

    def call(self, inputs):
        """
        Parameters
            inputs: list with two entries
        """

        # check shapes
        assert len(inputs) == 2, "inputs has to be len 2, found: %d" % len(inputs)
        vol = inputs[0]
        trf = inputs[1]

        # necessary for multi_gpu models...
        vol = K.reshape(vol, [-1, *self.inshape[0][1:]])
        trf = K.reshape(trf, [-1, *self.inshape[1][1:]])

        # convert matrix to warp field
        if self.is_affine:
            ncols = self.ndims + 1
            nrows = self.ndims
            if np.prod(trf.shape.as_list()[1:]) == (self.ndims + 1) ** 2:
                nrows += 1
            if len(trf.shape[1:]) == 1:
                trf = tf.reshape(trf, shape=(-1, nrows, ncols))
            if self.add_identity:
                trf += tf.eye(nrows, ncols, batch_shape=(tf.shape(trf)[0],))
            fun = lambda x: affine_to_dense_shift(x, vol.shape[1:-1], shift_center=self.shift_center)
            trf = tf.map_fn(fun, trf, dtype=tf.float32)

        # prepare location shift
        if self.indexing == 'xy':  # shift the first two dimensions
            trf_split = tf.split(trf, trf.shape[-1], axis=-1)
            trf_lst = [trf_split[1], trf_split[0], *trf_split[2:]]
            trf = tf.concat(trf_lst, -1)

        # map transform across batch
        if self.single_transform:
            fn = lambda x: self._single_transform([x, trf[0, :]])
            return tf.map_fn(fn, vol, dtype=tf.float32)
        else:
            return tf.map_fn(self._single_transform, [vol, trf], dtype=tf.float32)

    def _single_transform(self, inputs):
        return transform(inputs[0], inputs[1], interp_method=self.interp_method, fill_value=self.fill_value)

class ComposeTransform(tf.keras.layers.Layer):
    """
    Composes a single transform from a series of transforms.
    Supports both dense and affine transforms, and returns a dense transform unless all
    inputs are affine. The list of transforms to compose should be in the order in which
    they would be individually applied to an image. For example, given transforms A, B,
    and C, to compose a single transform T, where T(x) = C(B(A(x))), the appropriate
    function call is:
    T = ComposeTransform()([A, B, C])
    """

    def __init__(self, interp_method='linear', shift_center=True, indexing='ij', **kwargs):
        """
        Parameters:
            shape: Target shape of dense shift.
            interp_method: Interpolation method. Must be 'linear' or 'nearest'.
            shift_center: Shift grid to image center.
            indexing: Must be 'xy' or 'ij'.
        """
        self.interp_method = interp_method
        self.shift_center = shift_center
        self.indexing = indexing
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'interp_method': self.interp_method,
            'shift_center': self.shift_center,
            'indexing': self.indexing,
        })
        return config

    def build(self, input_shape, **kwargs):

        # sanity check on the inputs
        if not isinstance(input_shape, (list, tuple)):
            raise Exception('ComposeTransform must be called for a list of transforms.')
        if len(input_shape) < 2:
            raise ValueError('ComposeTransform input list size must be greater than 1.')

        # determine output transform type
        dense_shape = next((t for t in input_shape if not is_affine_shape(t[1:])), None)
        if dense_shape is not None:
            # extract shape information from the dense transform
            self.outshape = (input_shape[0], *dense_shape)
        else:
            # extract dimension information from affine
            ndims = input_shape[0][-1] - 1
            self.outshape = (input_shape[0], ndims, ndims + 1)

    def call(self, transforms):
        """
        Parameters:
            transforms: List of affine or dense transforms to compose.
        """
        compose_ = lambda trf: compose(trf, interp_method=self.interp_method,shift_center=self.shift_center, indexing=self.indexing)
        return tf.map_fn(compose_, transforms, dtype=tf.float32)

    def compute_output_shape(self, input_shape):
        return self.outshape

def compose(transforms, interp_method='linear', shift_center=True, indexing='ij'):
    """
    Compose a single transform from a series of transforms.
    Supports both dense and affine transforms, and returns a dense transform unless all
    inputs are affine. The list of transforms to compose should be in the order in which
    they would be individually applied to an image. For example, given transforms A, B,
    and C, to compose a single transform T, where T(x) = C(B(A(x))), the appropriate
    function call is:
    T = compose([A, B, C])
    Parameters:
        transforms: List of affine and/or dense transforms to compose.
        interp_method: Interpolation method. Must be 'linear' or 'nearest'.
        shift_center: Shift grid to image center.
        indexing: Must be 'xy' or 'ij'.
    Returns:
        Composed affine or dense transform.
    """
    if indexing != 'ij':
        raise ValueError('Compose transform only supports ij indexing')

    if len(transforms) < 2:
        raise ValueError('Compose transform list size must be greater than 1')

    def ensure_dense(trf, shape):
        if is_affine_shape(trf.shape):
            return affine_to_dense_shift(trf, shape, shift_center=shift_center, indexing=indexing)
        return trf

    def ensure_square_affine(matrix):
        if matrix.shape[-1] != matrix.shape[-2]:
            return make_square_affine(matrix)
        return matrix

    curr = transforms[-1]
    for nxt in reversed(transforms[:-1]):
        # check if either transform is dense
        found_dense = next((t for t in (nxt, curr) if not is_affine_shape(t.shape)), None)
        if found_dense is not None:
            # compose dense warps
            shape = found_dense.shape[:-1]
            nxt = ensure_dense(nxt, shape)
            curr = ensure_dense(curr, shape)
            curr = curr + transform(nxt, curr, interp_method=interp_method, indexing=indexing)
            #curr = transform(nxt, curr, interp_method=interp_method, indexing=indexing)

        else:
            # compose affines
            nxt = ensure_square_affine(nxt)
            curr = ensure_square_affine(curr)
            curr = tf.linalg.matmul(nxt, curr)[:-1]

    return curr

def make_square_affine(mat):
    """
    Converts a [N, N+1] affine matrix to square shape [N+1, N+1].
    Parameters:
        mat: Affine matrix of shape [..., N, N+1].
    """
    validate_affine_shape(mat.shape)
    bs = mat.shape[:-2]
    zeros = tf.zeros((*bs, 1, mat.shape[-2]), dtype=mat.dtype)
    one = tf.ones((*bs, 1, 1), dtype=mat.dtype)
    row = tf.concat((zeros, one), axis=-1)
    mat = tf.concat([mat, row], axis=-2)
    return mat

def affine_to_dense_shift(matrix, shape, shift_center=True, indexing='ij'):
    """
    Transforms an affine matrix to a dense location shift.
    Algorithm:
        1. Build and (optionally) shift grid to center of image.
        2. Apply affine matrix to each index.
        3. Subtract grid.
    Parameters:
        matrix: affine matrix of shape (N, N+1).
        shape: ND shape of the target warp.
        shift_center: Shift grid to image center.
        indexing: Must be 'xy' or 'ij'.
    Returns:
        Dense shift (warp) of shape (*shape, N).
    """
    import neurite as ne

    if isinstance(shape, (tf.compat.v1.Dimension, tf.TensorShape)):
        shape = shape.as_list()

    if matrix.dtype != 'float32':
        matrix = tf.cast(matrix, 'float32')

    # check input shapes
    ndims = len(shape)
    if matrix.shape[-1] != (ndims + 1):
        matdim = matrix.shape[-1] - 1
        raise ValueError(f'Affine ({matdim}D) does not match target shape ({ndims}D).')
    validate_affine_shape(matrix.shape)

    # list of volume ndgrid
    # N-long list, each entry of shape
    mesh = ne.utils.volshape_to_meshgrid(shape, indexing=indexing)
    mesh = [tf.cast(f, 'float32') for f in mesh]

    if shift_center:
        mesh = [mesh[f] - (shape[f] - 1) / 2 for f in range(len(shape))]

    # add an all-ones entry and transform into a large matrix
    flat_mesh = [ne.utils.flatten(f) for f in mesh]
    flat_mesh.append(tf.ones(flat_mesh[0].shape, dtype='float32'))
    mesh_matrix = tf.transpose(tf.stack(flat_mesh, axis=1))  # 4 x nb_voxels

    # compute locations
    loc_matrix = tf.matmul(matrix, mesh_matrix)  # N+1 x nb_voxels
    loc_matrix = tf.transpose(loc_matrix[:ndims, :])  # nb_voxels x N
    loc = tf.reshape(loc_matrix, list(shape) + [ndims])  # *shape x N

    # get shifts and return
    return loc - tf.stack(mesh, axis=ndims)

def validate_affine_shape(shape):
    """
    Validates whether the given input shape represents a valid affine matrix.
    Throws error if the shape is valid.
    Parameters:
        shape: List of integers of the form [..., N, N+1].
    """
    ndim = shape[-1] - 1
    actual = tuple(shape[-2:])
    if ndim not in (2, 3) or actual != (ndim, ndim + 1):
        raise ValueError(f'Affine matrix must be of shape (2, 3) or (3, 4), got {actual}.')

def is_affine_shape(shape):
    """
    Determins whether the given shape (single-batch) represents an
    affine matrix.
    Parameters:
        shape:  List of integers of the form [N, N+1], assuming an affine.
    """
    if len(shape) == 2 and shape[-1] != 1:
        validate_affine_shape(shape)
        return True
    return False

def transform(vol, loc_shift, interp_method='linear', indexing='ij', fill_value=None):
    """
    transform (interpolation N-D volumes (features) given shifts at each location in tensorflow
    Essentially interpolates volume vol at locations determined by loc_shift.
    This is a spatial transform in the sense that at location [x] we now have the data from,
    [x + shift] so we've moved data.
    Args:
        vol (Tensor): volume with size vol_shape or [*vol_shape, C]
            where C is the number of channels
        loc_shift: shift volume [*new_vol_shape, D] or [*new_vol_shape, C, D]
            where C is the number of channels, and D is the dimentionality len(vol_shape)
            If loc_shift is [*new_vol_shape, D], it applies to all channels of vol
        interp_method (default:'linear'): 'linear', 'nearest'
        indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian).
            In general, prefer to leave this 'ij'
        fill_value (default: None): value to use for points outside the domain.
            If None, the nearest neighbors will be used.
    Return:
        new interpolated volumes in the same size as loc_shift[0]
    Keyworks:
        interpolation, sampler, resampler, linear, bilinear
    """
    import neurite as ne
    # parse shapes.
    # location volshape, including channels if available
    loc_volshape = loc_shift.shape[:-1]
    if isinstance(loc_volshape, (tf.compat.v1.Dimension, tf.TensorShape)):
        loc_volshape = loc_volshape.as_list()

    # volume dimensions
    nb_dims = len(vol.shape) - 1
    is_channelwise = len(loc_volshape) == (nb_dims + 1)
    assert loc_shift.shape[-1] == nb_dims, \
        'Dimension check failed for ne.utils.transform(): {}D volume (shape {}) called ' \
        'with {}D transform'.format(nb_dims, vol.shape[:-1], loc_shift.shape[-1])

    # location should be mesh and delta
    mesh = ne.utils.volshape_to_meshgrid(loc_volshape, indexing=indexing)  # volume mesh
    loc = [tf.cast(mesh[d], 'float32') + loc_shift[..., d] for d in range(nb_dims)]

    # if channelwise location, then append the channel as part of the location lookup
    if is_channelwise:
        loc.append(tf.cast(mesh[-1], 'float32'))

    # test single
    return ne.utils.interpn(vol, loc, interp_method=interp_method, fill_value=fill_value)