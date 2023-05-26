"""
tensorflow/keras utilities for the neuron project

If you use this code, please cite 
Dalca AV, Guttag J, Sabuncu MR
Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
CVPR 2018

or for the transformation/integration functions:

Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
MICCAI 2018.

Contact: adalca [at] csail [dot] mit [dot] edu
License: GPLv3
"""

import sys

# third party
import numpy as np
import tensorflow
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import keras.initializers
from keras.layers import Layer, InputLayer, Input
from tensorflow.python.keras.engine import base_layer

from tensorflow.python.keras import backend
#from tensorflow.python import roll as _roll
# from tensorflow.python.keras.engine.base_layer import Node

# local
from . import utils
from .utils import transform, resize, integrate_vec, affine_to_shift


class Negate(Layer):
    """
    Keras Layer: negative of the input
    """

    def __init__(self, **kwargs):
        super(Negate, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Negate, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return -x

    def compute_output_shape(self, input_shape):
        return input_shape


class RescaleValues(Layer):
    """
    Very simple Keras layer to rescale data values (e.g. intensities) by fixed factor
    """

    def __init__(self, resize, **kwargs):
        self.resize = resize
        super(RescaleValues, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RescaleValues, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x * self.resize

    def compute_output_shape(self, input_shape):
        return input_shape


class Resize(Layer):
    """
    N-D Resize Tensorflow / Keras Layer
    Note: this is not re-shaping an existing volume, but resizing, like scipy's "Zoom"

    If you find this function useful, please cite:
        Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation,Dalca AV, Guttag J, Sabuncu MR
        CVPR 2018

    Since then, we've re-written the code to be generalized to any
    dimensions, and along the way wrote grid and interpolation functions
    """

    def __init__(self,
                 zoom_factor,
                 interp_method='linear',
                 **kwargs):
        """
        Parameters:
            interp_method: 'linear' or 'nearest'
                'xy' indexing will have the first two entries of the flow
                (along last axis) flipped compared to 'ij' indexing
        """
        self.zoom_factor = zoom_factor
        self.interp_method = interp_method
        self.ndims = None
        self.inshape = None
        super(Resize, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        input_shape should be an element of list of one inputs:
        input1: volume
                should be a *vol_shape x N
        """

        if isinstance(input_shape[0], (list, tuple)) and len(input_shape) > 1:
            raise Exception('Resize must be called on a list of length 1.')

        if isinstance(input_shape[0], (list, tuple)):
            input_shape = input_shape[0]

        # set up number of dimensions
        self.ndims = len(input_shape) - 2
        self.inshape = input_shape
        if not isinstance(self.zoom_factor, (list, tuple)):
            self.zoom_factor = [self.zoom_factor] * self.ndims
        else:
            assert len(self.zoom_factor) == self.ndims, \
                'zoom factor length {} does not match number of dimensions {}'\
                    .format(len(self.zoom_factor), self.ndims)

        # confirm built
        self.built = True

        super(Resize, self).build(input_shape)  # Be sure to call this somewhere!


    def call(self, inputs):
        """
        Parameters
            inputs: volume of list with one volume
        """

        # check shapes
        if isinstance(inputs, (list, tuple)):
            assert len(inputs) == 1, "inputs has to be len 1. found: %d" % len(inputs)
            vol = inputs[0]
        else:
            vol = inputs

        # necessary for multi_gpu models...
        vol = K.reshape(vol, [-1, *self.inshape[1:]])

        # map transform across batch
        return tf.map_fn(self._single_resize, vol, dtype=tf.float32)

    def compute_output_shape(self, input_shape):

        output_shape = [input_shape[0]]
        output_shape += [int(input_shape[1:-1][f] * self.zoom_factor[f]) for f in range(self.ndims)]
        output_shape += [input_shape[-1]]
        return tuple(output_shape)

    def _single_resize(self, inputs):
        return resize(inputs, self.zoom_factor, interp_method=self.interp_method)

# Zoom naming of resize, to match scipy's naming
Zoom = Resize


class MSE(Layer):
    """
    Keras Layer: mean squared error
    """

    def __init__(self, **kwargs):
        super(MSE, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MSE, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.mean(K.batch_flatten(K.square(x[0] - x[1])), -1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], )


#########################################################
# Vector fields and spatial transforms
#########################################################

class SpatialTransformer(Layer):
    """
    N-D Spatial Transformer Tensorflow / Keras Layer

    The Layer can handle both affine and dense transforms.
    Both transforms are meant to give a 'shift' from the current position.
    Therefore, a dense transform gives displacements (not absolute locations) at each voxel,
    and an affine transform gives the *difference* of the affine matrix from
    the identity matrix.

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
                 ident=True,
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
            ident (default True), if True adds the ident matrix to the affine translation matrix
        """
        self.interp_method = interp_method
        self.fill_value = fill_value
        self.ndims = None
        self.inshape = None
        self.single_transform = single_transform
        self.ident = ident

        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing

        super(self.__class__, self).__init__(**kwargs)

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
        # it's a 2D Tensor and shape == [N+1, N+1].
        #   [dense with N=1, which is the only one that could have a transform shape of 2, would be of size Mx1]
        self.is_affine = len(trf_shape) == 1 or \
                         (len(trf_shape) == 2 and all([trf_shape[0] == self.ndims, trf_shape[1] == self.ndims+1]))

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
        vol = tf.cast(vol, tf.float32)
        trf = tf.cast(trf, tf.float32)

        # necessary for multi_gpu models...
        vol = K.reshape(vol, [-1, *self.inshape[0][1:]])
        trf = K.reshape(trf, [-1, *self.inshape[1][1:]])

        # go from affine
        if self.is_affine:
            trf = tf.map_fn(lambda x: self._single_aff_to_shift(x, vol.shape[1:-1]), trf, dtype=tf.float32)

        # prepare location shift
        if self.indexing == 'xy':  # shift the first two dimensions
            trf_split = tf.split(trf, trf.shape[-1], axis=-1)
            trf_lst = [trf_split[1], trf_split[0], *trf_split[2:]]
            trf = tf.concat(trf_lst, -1)

        # map transform across batch
        if self.single_transform:
            fn = lambda x: self._single_transform([x, trf[0,:]])
            return tf.map_fn(fn, vol, dtype=tf.float32)
        else:
            return tf.map_fn(self._single_transform, [vol, trf], dtype=tf.float32)

    def _single_aff_to_shift(self, trf, volshape):
        if len(trf.shape) == 1:  # go from vector to matrix
            trf = tf.reshape(trf, [self.ndims, self.ndims + 1])

        if self.ident:
            # note this is unnecessarily extra graph since at every batch entry we have a tf.eye graph
            b = tf.eye(self.ndims + 1)[:self.ndims,:]  # add identity, only if trf could be non eye, hence affine is a shift from identitiy
            trf += b
        return affine_to_shift(trf, volshape, shift_center=True)

    # TODO: delete inputs[1] from the return statement
    def _single_transform(self, inputs):
        return transform(inputs[0], inputs[1], interp_method=self.interp_method, fill_value=self.fill_value)

    def get_config(self):
        """ __init__() is overwritten, need to override this method to enable model.to_json() for this layer"""
        config = super(self.__class__, self).get_config()
        config.update({'interp_method': self.interp_method,
                       'indexing': self.indexing,
                       'single_transform':self.single_transform,
                       'fill_value': self.fill_value,
                       'ident': self.ident})
        return config



class SpatialTransformer_orig(Layer):
    """
    ND spatial transformer layer
    Applies affine and dense transforms to images. A dense transform gives
    displacements (not absolute locations) at each voxel.
    If you find this layer useful, please cite:
      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.
    Originally, this code was based on voxelmorph code, which
    was in turn transformed to be dense with the help of (affine) STN code
    via https://github.com/kevinzakka/spatial-transformer-network.
    Since then, we've re-written the code to be generalized to any
    dimensions, and along the way wrote grid and interpolation functions.
    """

    def __init__(self,
                 interp_method='linear',
                 indexing='ij',
                 single_transform=False,
                 fill_value=None,
                 shift_center=True,
                 **kwargs):
        """
        Parameters:
            interp_method: Interpolation method. Must be 'linear' or 'nearest'.
            indexing: Must be 'ij' (matrix) or 'xy' (cartesian). 'xy' indexing will
                have the first two entries of the flow (along last axis) flipped
                compared to 'ij' indexing.
            single_transform: Use single transform for the entire image batch.
            fill_value: Value to use for points sampled outside the domain.
                If None, the nearest neighbors will be used.
            shift_center: Shift grid to image center when converting affine
                transforms to dense transforms.
        """
        self.interp_method = interp_method
        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing
        self.single_transform = single_transform
        self.fill_value = fill_value
        self.shift_center = shift_center
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'interp_method': self.interp_method,
            'indexing': self.indexing,
            'single_transform': self.single_transform,
            'fill_value': self.fill_value,
            'shift_center': self.shift_center,
        })
        return config

    def build(self, input_shape):

        # sanity check on input list
        if len(input_shape) > 2:
            raise ValueError('Spatial Transformer must be called on a list of length 2: '
                             'first argument is the image, second is the transform.')

        # set up number of dimensions
        self.ndims = len(input_shape[0]) - 2
        self.imshape = input_shape[0][1:]
        self.trfshape = input_shape[1][1:]
        from src.ext.neuron.neuron import utils
        self.is_affine = utils.is_affine_shape(input_shape[1][1:])

        # make sure inputs are reasonable shapes
        if self.is_affine:
            expected = (self.ndims, self.ndims + 1)
            actual = tuple(self.trfshape[-2:])
            if expected != actual:
                raise ValueError(f'Expected {expected} affine matrix, got {actual}.')
        else:
            image_shape = tuple(self.imshape[:-1])
            dense_shape = tuple(self.trfshape[:-1])
            if image_shape != dense_shape:
                raise ValueError('Dense transform must match image shape, got '
                                 f'got {image_shape} and {dense_shape}.')

        # confirm built
        self.built = True

    def call(self, inputs):
        """
        Parameters
            inputs: List of [img, trf], where img is the ND moving image and trf
            is either a dense warp of shape [B, D1, ..., DN, N] or an affine matrix
            of shape [B, N, N+1].
        """

        # necessary for multi-gpu models
        vol = K.reshape(inputs[0], (-1, *self.imshape))
        trf = K.reshape(inputs[1], (-1, *self.trfshape))

        # convert affine matrix to warp field
        if self.is_affine:
            fun = lambda x: utils.affine_to_dense_shift(x, vol.shape[1:-1],
                                                        shift_center=self.shift_center,
                                                        indexing=self.indexing)
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
        return utils.transform(inputs[0], inputs[1], interp_method=self.interp_method,
                               fill_value=self.fill_value)

class VecInt(Layer):
    """
    Vector Integration Layer

    Enables vector integration via several methods
    (ode or quadrature for time-dependent vector fields,
    scaling and squaring for stationary fields)

    If you find this function useful, please cite:
      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.
    """

    def __init__(self, indexing='ij', method='ss', int_steps=7, out_time_pt=1,
                 ode_args=None,
                 odeint_fn=None, **kwargs):
        """
        Parameters:
            method can be any of the methods in neuron.utils.integrate_vec
            indexing can be 'xy' (switches first two dimensions) or 'ij'
            int_steps is the number of integration steps
            out_time_pt is time point at which to output if using odeint integration
        """

        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing
        self.method = method
        self.int_steps = int_steps
        self.inshape = None
        self.out_time_pt = out_time_pt
        self.odeint_fn = odeint_fn  # if none then will use a tensorflow function
        self.ode_args = ode_args
        if ode_args is None:
            self.ode_args = {'rtol':1e-6, 'atol':1e-12}
        super(self.__class__, self).__init__(**kwargs)

    def build(self, input_shape):
        # confirm built
        self.built = True

        trf_shape = input_shape
        if isinstance(input_shape[0], (list, tuple)):
            trf_shape = input_shape[0]
        self.inshape = trf_shape

        if trf_shape[-1] != len(trf_shape) - 2:
            raise Exception('transform ndims %d does not match expected ndims %d' \
                % (trf_shape[-1], len(trf_shape) - 2))

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        loc_shift = inputs[0]

        # necessary for multi_gpu models...
        loc_shift = K.reshape(loc_shift, [-1, *self.inshape[1:]])
        if hasattr(inputs[0], '_keras_shape'):
            loc_shift._keras_shape = inputs[0]._keras_shape

        # prepare location shift
        if self.indexing == 'xy':  # shift the first two dimensions
            loc_shift_split = tf.split(loc_shift, loc_shift.shape[-1], axis=-1)
            loc_shift_lst = [loc_shift_split[1], loc_shift_split[0], *loc_shift_split[2:]]
            loc_shift = tf.concat(loc_shift_lst, -1)

        if len(inputs) > 1:
            assert self.out_time_pt is None, 'out_time_pt should be None if providing batch_based out_time_pt'

        # map transform across batch
        out = tf.map_fn(self._single_int, [loc_shift] + inputs[1:], dtype=tf.float32)
        if hasattr(inputs[0], '_keras_shape'):
            out._keras_shape = inputs[0]._keras_shape
        return out

    def _single_int(self, inputs):

        vel = inputs[0]
        out_time_pt = self.out_time_pt
        if len(inputs) == 2:
            out_time_pt = inputs[1]
        return integrate_vec(vel, method=self.method,
                      nb_steps=self.int_steps,
                      ode_args=self.ode_args,
                      out_time_pt=out_time_pt,
                      odeint_fn=self.odeint_fn)


# full wording.
VecIntegration = VecInt


#########################################################
# Sparse layers
#########################################################

class SpatiallySparse_Dense(Layer):
    """
    Spatially-Sparse Dense Layer (great name, huh?)
    This is a Densely connected (Fully connected) layer with sparse observations.

    # layer can (and should) be used when going from vol to embedding *and* going back.
    # it will account for the observed variance and maintain the same weights

    # if going vol --> enc:
    # tensor inputs should be [vol, mask], and output will be a encoding tensor enc
    # if going enc --> vol:
    # tensor inputs should be [enc], and output will be vol
    """

    def __init__(self, input_shape, output_len, use_bias=False,
                 kernel_initializer='RandomNormal',
                 bias_initializer='RandomNormal', **kwargs):
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.output_len = output_len
        self.cargs = 0
        self.use_bias = use_bias
        self.orig_input_shape = input_shape  # just the image size
        super(SpatiallySparse_Dense, self).__init__(**kwargs)

    def build(self, input_shape):



        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='mult-kernel',
                                    shape=(np.prod(self.orig_input_shape),
                                           self.output_len),
                                    initializer=self.kernel_initializer,
                                    trainable=True)

        M = K.reshape(self.kernel, [-1, self.output_len])  # D x d
        mt = K.transpose(M) # d x D
        mtm_inv = tf.matrix_inverse(K.dot(mt, M))  # d x d
        self.W = K.dot(mtm_inv, mt) # d x D

        if self.use_bias:
            self.bias = self.add_weight(name='bias-kernel',
                                        shape=(self.output_len, ),
                                        initializer=self.bias_initializer,
                                        trainable=True)

        # self.sigma_sq = self.add_weight(name='bias-kernel',
        #                                 shape=(1, ),
        #                                 initializer=self.initializer,
        #                                 trainable=True)

        super(SpatiallySparse_Dense, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, args):

        if not isinstance(args, (list, tuple)):
            args = [args]
        self.cargs = len(args)

        # flatten
        if len(args) == 2:  # input y, m
            # get inputs
            y, y_mask = args
            a_fact = int(y.get_shape().as_list()[-1] / y_mask.get_shape().as_list()[-1])
            y_mask = K.repeat_elements(y_mask, a_fact, -1)
            y_flat = K.batch_flatten(y)  # N x D
            y_mask_flat = K.batch_flatten(y_mask)  # N x D

            # prepare switching matrix
            W = self.W # d x D

            w_tmp = K.expand_dims(W, 0)  # 1 x d x D
            Wo = K.permute_dimensions(w_tmp, [0, 2, 1]) * K.expand_dims(y_mask_flat, -1)  # N x D x d
            WoT = K.permute_dimensions(Wo, [0, 2, 1])    # N x d x D
            WotWo_inv = tf.matrix_inverse(K.batch_dot(WoT, Wo))  # N x d x d
            pre = K.batch_dot(WotWo_inv, WoT) # N x d x D
            res = K.batch_dot(pre, y_flat)  # N x d

            if self.use_bias:
                res += K.expand_dims(self.bias, 0)

        else:
            x_data = args[0]
            shape = K.shape(x_data)

            x_data = K.batch_flatten(x_data)  # N x d

            if self.use_bias:
                x_data -= self.bias

            res = K.dot(x_data, self.W)

            # reshape
            # Here you can mix integers and symbolic elements of `shape`
            pool_shape = tf.stack([shape[0], *self.orig_input_shape])
            res = K.reshape(res, pool_shape)

        return res

    def compute_output_shape(self, input_shape):
        # print(self.cargs, input_shape, self.output_len, self.orig_input_shape)
        if self.cargs == 2:
            return (input_shape[0][0], self.output_len)
        else:
            return (input_shape[0], *self.orig_input_shape)


#########################################################
# "Local" layers -- layers with parameters at each voxel
#########################################################

class LocalBias(Layer):
    """
    Local bias layer: each pixel/voxel has its own bias operation (one parameter)
    out[v] = in[v] + b
    """

    def __init__(self, my_initializer='RandomNormal', biasmult=1.0, **kwargs):
        self.initializer = my_initializer
        self.biasmult = biasmult
        super(LocalBias, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=input_shape[1:],
                                      initializer=self.initializer,
                                      trainable=True)
        super(LocalBias, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x + self.kernel * self.biasmult  # weights are difference from input

    def compute_output_shape(self, input_shape):
        return input_shape


class LocalLinear(Layer):
    """
    Local linear layer: each pixel/voxel has its own linear operation (two parameters)
    out[v] = a * in[v] + b
    """

    def __init__(self, initializer='RandomNormal', **kwargs):
        self.initializer = initializer
        super(LocalLinear, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.mult = self.add_weight(name='mult-kernel',
                                      shape=input_shape[1:],
                                      initializer=self.initializer,
                                      trainable=True)
        self.bias = self.add_weight(name='bias-kernel',
                                      shape=input_shape[1:],
                                      initializer=self.initializer,
                                      trainable=True)
        super(LocalLinear, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x * self.mult + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape




class LocalCrossLinear(tensorflow.keras.layers.Layer):
    """
    Local cross mult layer

    input: [batch_size, *vol_size, nb_feats_1]
    output: [batch_size, *vol_size, nb_feats_2]

    at each spatial voxel, there is a different linear relation learned.
    """

    def __init__(self, output_features,
                 mult_initializer=None,
                 bias_initializer=None,
                 mult_regularizer=None,
                 bias_regularizer=None,
                 use_bias=True,
                 **kwargs):

        self.output_features = output_features
        self.mult_initializer = mult_initializer
        self.bias_initializer = bias_initializer
        self.mult_regularizer = mult_regularizer
        self.bias_regularizer = bias_regularizer
        self.use_bias = use_bias

        super(LocalCrossLinear, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        mult_shape = [1] + list(input_shape)[1:] + [self.output_features]


        # verify initializer
        if self.mult_initializer is None:
            mean = 1/input_shape[-1]
            stddev = 0.01
            self.mult_initializer = keras.initializers.RandomNormal(mean=mean, stddev=stddev)

        self.mult = self.add_weight(name='mult-kernel',
                                      shape=mult_shape,
                                      initializer=self.mult_initializer,
                                      regularizer=self.mult_regularizer,
                                      trainable=True)

        if self.use_bias:
            if self.bias_initializer is None:
                mean = 1/input_shape[-1]
                stddev = 0.01
                self.bias_initializer = keras.initializers.RandomNormal(mean=mean, stddev=stddev)

            bias_shape = [1] + list(input_shape)[1:-1] + [self.output_features]
            self.bias = self.add_weight(name='bias-kernel',
                                          shape=bias_shape,
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          trainable=True)
        super(LocalCrossLinear, self).build(input_shape)

    def call(self, x):
        map_fn = lambda z: self._single_matmul(z, self.mult[0, ...])
        y = tf.stack(tf.map_fn(map_fn, x, dtype=tf.float32), 0)

        if self.use_bias:
            y = y + self.bias

        return y

    def _single_matmul(self, x, mult):
        x = K.expand_dims(x, -2)
        y = tf.matmul(x, mult)[...,0,:]
        return y

    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape)[:-1] + [self.output_features])



class LocalCrossLinearTrf(keras.layers.Layer):
    """
    Local cross mult layer with transform

    input: [batch_size, *vol_size, nb_feats_1]
    output: [batch_size, *vol_size, nb_feats_2]

    at each spatial voxel, there is a different linear relation learned.
    """

    def __init__(self, output_features,
                 mult_initializer=None,
                 bias_initializer=None,
                 mult_regularizer=None,
                 bias_regularizer=None,
                 use_bias=True,
                 trf_mult=1,
                 **kwargs):

        self.output_features = output_features
        self.mult_initializer = mult_initializer
        self.bias_initializer = bias_initializer
        self.mult_regularizer = mult_regularizer
        self.bias_regularizer = bias_regularizer
        self.use_bias = use_bias
        self.trf_mult = trf_mult
        self.interp_method = 'linear'

        super(LocalCrossLinearTrf, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        mult_shape = list(input_shape)[1:] + [self.output_features]
        ndims = len(list(input_shape)[1:-1])


        # verify initializer
        if self.mult_initializer is None:
            mean = 1/input_shape[-1]
            stddev = 0.01
            self.mult_initializer = keras.initializers.RandomNormal(mean=mean, stddev=stddev)

        self.mult = self.add_weight(name='mult-kernel',
                                      shape=mult_shape,
                                      initializer=self.mult_initializer,
                                      regularizer=self.mult_regularizer,
                                      trainable=True)

        self.trf = self.add_weight(name='def-kernel',
                                      shape=mult_shape + [ndims],
                                      initializer=keras.initializers.RandomNormal(mean=0, stddev=0.001),
                                      trainable=True)

        if self.use_bias:
            if self.bias_initializer is None:
                mean = 1/input_shape[-1]
                stddev = 0.01
                self.bias_initializer = keras.initializers.RandomNormal(mean=mean, stddev=stddev)

            bias_shape = list(input_shape)[1:-1] + [self.output_features]
            self.bias = self.add_weight(name='bias-kernel',
                                          shape=bias_shape,
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          trainable=True)

        super(LocalCrossLinearTrf, self).build(input_shape)

    def call(self, x):


        # for each element in the batch
        y = tf.map_fn(self._single_batch_trf, x, dtype=tf.float32)

        return y

    def _single_batch_trf(self, vol):
        # vol should be vol_shape + [nb_features]
        # self.trf should be vol_shape + [nb_features] + [ndims]

        vol_shape = vol.shape.as_list()
        nb_input_dims = vol_shape[-1]

        # this is inefficient...
        new_vols = [None] * self.output_features
        for j in range(self.output_features):
            new_vols[j] = tf.zeros(vol_shape[:-1], dtype=tf.float32)
            for i in range(nb_input_dims):
                trf_vol = transform(vol[..., i], self.trf[..., i, j, :] * self.trf_mult, interp_method=self.interp_method)
                trf_vol = tf.reshape(trf_vol, vol_shape[:-1])
                new_vols[j] += trf_vol * self.mult[..., i, j]

                if self.use_bias:
                    new_vols[j] += self.bias[..., j]

        return tf.stack(new_vols, -1)


    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape)[:-1] + [self.output_features])


class LocalParamLayer(Layer):
    """
    Local Parameter layer: each pixel/voxel has its own parameter (one parameter)
    out[v] = b

    using code from
    https://github.com/YerevaNN/R-NET-in-Keras/blob/master/layers/SharedWeight.py
    and
    https://github.com/keras-team/keras/blob/ee02d256611b17d11e37b86bd4f618d7f2a37d84/keras/engine/input_layer.py
    """

    def __init__(self,
                 shape,
                 my_initializer='RandomNormal',
                 dtype=None,
                 name=None,
                 mult=1.0,
                 **kwargs):


        # some input checking
        if not name:
            prefix = 'local_param'
            name = prefix + '_' + str(backend.get_uid(prefix))

        if not dtype:
            dtype = backend.floatx()

        self.shape = [1, *shape]
        self.my_initializer = my_initializer
        self.mult = mult

        if not name:
            prefix = 'param'
            name = '%s_%d' % (prefix, K.get_uid(prefix))
        Layer.__init__(self, name=name, **kwargs)

        # Create a trainable weight variable for this layer.
        with K.name_scope(self.name):
            self.kernel = self.add_weight(name='kernel',
                                            shape=shape,
                                            initializer=self.my_initializer,
                                            dtype=dtype,
                                            trainable=True)

        # prepare output tensor, which is essentially the kernel.
        output_tensor = K.expand_dims(self.kernel, 0) * self.mult
        output_tensor._keras_shape = self.shape
        output_tensor._uses_learning_phase = False
        output_tensor._keras_history = base_layer.KerasHistory(self, 0, 0)
        output_tensor._batch_input_shape = self.shape

        self.trainable = True
        self.built = True
        self.is_placeholder = False

        # create new node
        tensorflow.python.keras.engine.base_layer.node_module.Node(self,
            inbound_layers=[],
            node_indices=[],
            tensor_indices=[],
            input_tensors=[],
            output_tensors=[output_tensor],
            input_masks=[],
            output_masks=[None],
            input_shapes=[],
            output_shapes=self.shape)

    def get_config(self):
        config = {
                'dtype': self.dtype,
                'sparse': self.sparse,
                'name': self.name
        }
        return config


def LocalParam(    # pylint: disable=invalid-name
        shape,
        batch_size=None,
        name=None,
        dtype=None,
        **kwargs):
    """
    `LocalParam()` is used to instantiate a Keras tensor.
    A Keras tensor is a tensor object from the underlying backend
    (Theano or TensorFlow), which we augment with certain
    attributes that allow us to build a Keras model
    just by knowing the inputs and outputs of the model.
    For instance, if a, b and c are Keras tensors,
    it becomes possible to do:
    `model = Model(input=[a, b], output=c)`
    The added Keras attribute is:
            `_keras_history`: Last layer applied to the tensor.
                    the entire layer graph is retrievable from that layer,
                    recursively.
    Arguments:
            shape: A shape tuple (integers), not including the batch size.
                    For instance, `shape=(32,)` indicates that the expected input
                    will be batches of 32-dimensional vectors. Elements of this tuple
                    can be None; 'None' elements represent dimensions where the shape is
                    not known.
            batch_size: optional static batch size (integer).
            name: An optional name string for the layer.
                    Should be unique in a model (do not reuse the same name twice).
                    It will be autogenerated if it isn't provided.
            dtype: The data type expected by the input, as a string
                    (`float32`, `float64`, `int32`...)
            **kwargs: deprecated arguments support.
    Returns:
        A `tensor`.
    Example:
    ```python
    # this is a logistic regression in Keras
    x = Input(shape=(32,))
    y = Dense(16, activation='softmax')(x)
    model = Model(x, y)
    ```
    Note that even if eager execution is enabled,
    `Input` produces a symbolic tensor (i.e. a placeholder).
    This symbolic tensor can be used with other
    TensorFlow ops, as such:
    ```python
    x = Input(shape=(32,))
    y = tf.square(x)
    ```
    Raises:
        ValueError: in case of invalid arguments.
    """
    input_layer = LocalParamLayer(shape, name=name, dtype=dtype)

    # Return tensor including `_keras_history`.
    # Note that in this case train_output and test_output are the same pointer.
    outputs = input_layer._inbound_nodes[0].output_tensors
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs


##########################################
## Stream layers
##########################################


class MeanStream(Layer):
    """
    Maintain stream of data mean.

    cap refers to mainting an approximation of up to that number of subjects -- that is,
    any incoming datapoint will have at least 1/cap weight.
    """

    def __init__(self, cap=100, **kwargs):
        self.cap = K.variable(cap, dtype='float32')
        super(MeanStream, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create mean and count
        # These are weights because just maintaining variables don't get saved with the model, and we'd like
        # to have these numbers saved when we save the model.
        # But we need to make sure that the weights are untrainable.
        self.mean = self.add_weight(name='mean',
                                      shape=input_shape[1:],
                                      initializer='zeros',
                                      trainable=False)
        self.count = self.add_weight(name='count',
                                      shape=[1],
                                      initializer='zeros',
                                      trainable=False)

        # self.mean = K.zeros(input_shape[1:], name='mean')
        # self.count = K.variable(0.0, name='count')
        super(MeanStream, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        # get new mean and count
        this_bs_int = K.shape(x)[0]
        new_mean, new_count = _mean_update(self.mean, self.count, x, self.cap)

        # update op
        updates = [(self.count, new_count), (self.mean, new_mean)]
        self.add_update(updates, x)

        # prep for broadcasting :(
        p = tf.concat((K.reshape(this_bs_int, (1,)), K.shape(self.mean)), 0)
        z = tf.ones(p)

        # the first few 1000 should not matter that much towards this cost
        return K.minimum(1., new_count/self.cap) * (z * K.expand_dims(new_mean, 0))

    def compute_output_shape(self, input_shape):
        return input_shape


class CovStream(Layer):
    """
    Maintain stream of data mean.

    cap refers to mainting an approximation of up to that number of subjects -- that is,
    any incoming datapoint will have at least 1/cap weight.
    """

    def __init__(self, cap=100, **kwargs):
        self.cap = K.variable(cap, dtype='float32')
        super(CovStream, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create mean, cov and and count
        # See note in MeanStream.build()
        self.mean = self.add_weight(name='mean',
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=False)
        v = np.prod(input_shape[1:])
        self.cov = self.add_weight(name='cov',
                                 shape=[v, v],
                                 initializer='zeros',
                                 trainable=False)
        self.count = self.add_weight(name='count',
                                      shape=[1],
                                      initializer='zeros',
                                      trainable=False)

        super(CovStream, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        x_orig = x

        # x reshape
        this_bs_int = K.shape(x)[0]
        this_bs = tf.cast(this_bs_int, 'float32')  # this batch size
        prev_count = self.count
        x = K.batch_flatten(x)  # B x N

        # update mean
        new_mean, new_count = _mean_update(self.mean, self.count, x, self.cap)

        # new C update. Should be B x N x N
        x = K.expand_dims(x, -1)
        C_delta = K.batch_dot(x, K.permute_dimensions(x, [0, 2, 1]))

        # update cov
        prev_cap = K.minimum(prev_count, self.cap)
        C = self.cov * (prev_cap - 1) + K.sum(C_delta, 0)
        new_cov = C / (prev_cap + this_bs - 1)

        # updates
        updates = [(self.count, new_count), (self.mean, new_mean), (self.cov, new_cov)]
        self.add_update(updates, x_orig)

        # prep for broadcasting :(
        p = tf.concat((K.reshape(this_bs_int, (1,)), K.shape(self.cov)), 0)
        z = tf.ones(p)

        return K.minimum(1., new_count/self.cap) * (z * K.expand_dims(new_cov, 0))

    def compute_output_shape(self, input_shape):
        v = np.prod(input_shape[1:])
        return (input_shape[0], v, v)


def _mean_update(pre_mean, pre_count, x, pre_cap=None):

    # compute this batch stats
    this_sum = tf.reduce_sum(x, 0)
    this_bs = tf.cast(K.shape(x)[0], 'float32')  # this batch size

    # increase count and compute weights
    new_count = pre_count + this_bs
    alpha = this_bs/K.minimum(new_count, pre_cap)

    # compute new mean. Note that once we reach self.cap (e.g. 1000), the 'previous mean' matters less
    new_mean = pre_mean * (1-alpha) + (this_sum/this_bs) * alpha

    return (new_mean, new_count)

##########################################
## FFT Layers
##########################################

class FFT(Layer):
    """
    fft layer, assuming the real/imag are input/output via two features
    Input: tf.complex of size [batch_size, ..., nb_feats]
    Output: tf.complex of size [batch_size, ..., nb_feats]
    """

    def __init__(self, **kwargs):
        super(FFT, self).__init__(**kwargs)

    def build(self, input_shape):
        # some input checking
        self.ndims = len(input_shape) - 2
        assert self.ndims in [1, 2, 3], 'only 1D, 2D or 3D supported'

        # super
        super(FFT, self).build(input_shape)

    def call(self, inputx):

        if not inputx.dtype in [tf.complex64, tf.complex128]:
            print('Warning: inputx is not complex. Converting.', file=sys.stderr)

            # if inputx is float, this will assume 0 imag channel
            inputx = tf.cast(inputx, tf.complex64)

        # get the right fft
        if self.ndims == 1:
            fft = tf.fft
        elif self.ndims == 2:
            fft = tf.fft2d
        else:
            fft = tf.fft3d

        perm_dims = [0, self.ndims + 1] + list(range(1, self.ndims + 1))
        invert_perm_ndims = [0] + list(range(2, self.ndims + 2)) + [1]

        perm_inputx = K.permute_dimensions(inputx, perm_dims)  # [batch_size, nb_features, *vol_size]
        fft_inputx = fft(perm_inputx)
        return K.permute_dimensions(fft_inputx, invert_perm_ndims)

    def compute_output_shape(self, input_shape):
        return input_shape


class IFFT(Layer):
    """
    ifft layer, assuming the real/imag are input/output via two features
    Input: tf.complex of size [batch_size, ..., nb_feats]
    Output: tf.complex of size [batch_size, ..., nb_feats]
    """

    def __init__(self, **kwargs):
        super(IFFT, self).__init__(**kwargs)

    def build(self, input_shape):
        # some input checking
        self.ndims = len(input_shape) - 2
        assert self.ndims in [1, 2, 3], 'only 1D, 2D or 3D supported'

        # super
        super(IFFT, self).build(input_shape)

    def call(self, inputx):

        if not inputx.dtype in [tf.complex64, tf.complex128]:
            print('Warning: inputx is not complex. Converting.', file=sys.stderr)

            # if inputx is float, this will assume 0 imag channel
            inputx = tf.cast(inputx, tf.complex64)

        # get the right fft
        if self.ndims == 1:
            ifft = tf.ifft
        elif self.ndims == 2:
            ifft = tf.ifft2d
        else:
            ifft = tf.ifft3d

        perm_dims = [0, self.ndims + 1] + list(range(1, self.ndims + 1))
        invert_perm_ndims = [0] + list(range(2, self.ndims + 2)) + [1]

        perm_inputx = K.permute_dimensions(inputx, perm_dims)  # [batch_size, nb_features, *vol_size]
        ifft_inputx = ifft(perm_inputx)
        return K.permute_dimensions(ifft_inputx, invert_perm_ndims)

    def compute_output_shape(self, input_shape):
        return input_shape


class ComplexToChannels(Layer):

    def __init__(self, **kwargs):
        super(ComplexToChannels, self).__init__(**kwargs)

    def build(self, input_shape):
        # super
        super(ComplexToChannels, self).build(input_shape)

    def call(self, inputx):

        assert inputx.dtype in [tf.complex64, tf.complex128], 'inputx is not complex.'

        return tf.concat([tf.real(inputx), tf.imag(inputx)], -1)

    def compute_output_shape(self, input_shape):
        i_s = list(input_shape)
        i_s[-1] *= 2
        return tuple(i_s)


class ChannelsToComplex(Layer):

    def __init__(self, **kwargs):
        super(ChannelsToComplex, self).__init__(**kwargs)

    def build(self, input_shape):
        # super
        super(ChannelsToComplex, self).build(input_shape)

    def call(self, inputx):
        nb_channels = inputx.shape[-1] // 2
        return tf.complex(inputx[...,:nb_channels], inputx[...,nb_channels:])

    def compute_output_shape(self, input_shape):
        i_s = list(input_shape)
        i_s[-1] = i_s[-1] // 2
        return tuple(i_s)


class FFTShift(Layer):
    """
    fftshift for keras tensors (so only inner dimensions get shifted)

    modified from
    https://gist.github.com/Gurpreetsingh9465/f76cc9e53107c29fd76515d64c294d3f

    Shift the zero-frequency component to the center of the spectrum.
    This function swaps half-spaces for all axes listed (defaults to all).
    Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.
    Parameters
    ----------
    x : array_like, Tensor
        Input array.
    axes : int or shape tuple, optional
        Axes over which to shift.  Default is None, which shifts all axes.
    Returns
    -------
    y : Tensor.
    """

    def __init__(self, axes=None, **kwargs):
        self.axes = axes
        super(FFTShift, self).__init__(**kwargs)

    def build(self, input_shape):
        # some input checking
        self.ndims = len(input_shape) - 2
        assert self.ndims in [1, 2, 3], 'only 1D, 2D or 3D supported'

        # super
        super(FFTShift, self).build(input_shape)

    def call(self, x):
        axes = self.axes
        if axes is None:
            axes = tuple(range(K.ndim(x)))
            shift = [0] + [dim // 2 for dim in x.shape] + [0]
        elif isinstance(axes, int):
            shift = x.shape[axes] // 2
        else:
            shift = [x.shape[ax] // 2 for ax in axes]

        return _roll(x, shift, axes)

    def compute_output_shape(self, input_shape):
        return input_shape


class IFFTShift(Layer):
    """
    ifftshift for keras tensors (so only inner dimensions get shifted)

    modified from
    https://gist.github.com/Gurpreetsingh9465/f76cc9e53107c29fd76515d64c294d3f

    The inverse of `fftshift`. Although identical for even-length `x`, the
    functions differ by one sample for odd-length `x`.
    Parameters
    ----------
    x : array_like, Tensor.
    axes : int or shape tuple, optional
        Axes over which to calculate.  Defaults to None, which shifts all axes.
    Returns
    -------
    y : Tensor.
    """

    def __init__(self, axes=None, **kwargs):
        self.axes = axes
        super(IFFTShift, self).__init__(**kwargs)

    def build(self, input_shape):
        # some input checking
        self.ndims = len(input_shape) - 2
        assert self.ndims in [1, 2, 3], 'only 1D, 2D or 3D supported'

        # super
        super(IFFTShift, self).build(input_shape)

    def call(self, x):
        axes = self.axes
        if axes is None:
            axes = tuple(range(K.ndim(x)))
            shift = [0] + [-(dim // 2) for dim in x.shape.as_list()[1:-1]] + [0]
        elif isinstance(axes, int):
            shift = -(x.shape[axes] // 2)
        else:
            shift = [-(x.shape[ax] // 2) for ax in axes]

        return _roll(x, shift, axes)

    def compute_output_shape(self, input_shape):
        return input_shape




##########################################
## Stochastic Sampling layers
##########################################

class SampleNormalLogVar(Layer):
    """
    Keras Layer: Gaussian sample given mean and log_variance

    inputs: list of Tensors [mu, log_var]
    outputs: Tensor sample from N(mu, sigma^2)
    """

    def __init__(self, **kwargs):
        super(SampleNormalLogVar, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SampleNormalLogVar, self).build(input_shape)

    def call(self, x):
        return self._sample(x)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def _sample(self, args):
        """
        sample from a normal distribution

        args should be [mu, log_var], where log_var is the log of the squared sigma

        This is probably equivalent to
            K.random_normal(shape, args[0], exp(args[1]/2.0))
        """
        mu, log_var = args

        # sample from N(0, 1)
        noise = tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

        # make it a sample from N(mu, sigma^2)
        z = mu + tf.exp(log_var/2.0) * noise
        return z
