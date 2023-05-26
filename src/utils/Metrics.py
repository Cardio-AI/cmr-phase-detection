


import numpy as np
import tensorflow as tf

class Meandiff:

    def __init__(self, apply_sum=True, apply_average=True):
        self.__name__ = 'meandiff_sum_{}_avg_{}'.format(apply_sum, apply_average)
        self.apply_sum=apply_sum
        self.apply_average = apply_average

    def __call__(self, y_true, y_pred, **kwargs):
        return  meandiff(y_true, y_pred, apply_sum=self.apply_sum, apply_average=self.apply_average)

def meandiff( y_true, y_pred, apply_sum=True, apply_average=True):

    """
    Average over the batches
    the sum of the absolute difference between two arrays
    y_true and y_pred are one-hot vectors with the following shape
    batchsize * timesteps * phase classes
    e.g.: 4 * 36 * 5
    First for gt and pred:
    - get the timesteps per phase with the highest probability
    - get the absolute difference between gt and pred
    (- later we can slice each patient by the max value in the corresponding gt indices)
    - sum the diff per entity
    - calc the mean over all examples

    Parameters
    ----------
    y_true :
    y_pred :

    Returns tf.float32 scalar
    -------

    """

    y_true, y_len_msk = tf.unstack(y_true,2,axis=1)
    y_pred, _ = tf.unstack(y_pred,2,axis=1)

    y_true = tf.cast(tf.convert_to_tensor(y_true), tf.float32)
    y_pred = tf.cast(tf.convert_to_tensor(y_pred), tf.float32)
    y_len_msk = tf.cast(tf.convert_to_tensor(y_len_msk), tf.float32)

    # b, 36, 5
    temp_pred = y_pred * y_len_msk
    temp_gt = y_true * y_len_msk

    # get the original lengths of each mask in the current batch
    # b, 1
    y_len = tf.cast(tf.reduce_sum(y_len_msk[:,:,0], axis=1),dtype=tf.int32)#

    #print('y_len shape: {}'.format(y_len.shape))
    # returns b, 5,
    gt_idx = tf.cast(tf.math.argmax(temp_gt, axis=1), dtype=tf.int32)
    pred_idx = tf.cast(tf.math.argmax(temp_pred, axis=1), dtype=tf.int32)
    filled_length = tf.repeat(tf.expand_dims(y_len,axis=1),5,axis=1)

    # b, 5, 3
    stacked = tf.stack([gt_idx, pred_idx, filled_length], axis=-1)

    # sum the error per entity, and calc the mean over the batches
    # for each batch ==> 5, 3 in stacked
    diffs = tf.map_fn(lambda x: get_min_dist_for_list(x), stacked, dtype=tf.int32)
    if apply_sum: diffs = tf.cast(tf.reduce_sum(diffs, axis=1),tf.float32)
    if apply_average: diffs = tf.reduce_mean(diffs)
    diffs = tf.cast(diffs, tf.float32)
    #tf.math.greater_equal(diffs, 0), 'distance cant be smaller than 0'
    return diffs

@tf.function
def get_min_dist_for_list(vals):
    # vals has the shape 5, 3
    # for each phase tuple (gt,pred,length)
    return tf.map_fn(lambda x :get_min_distance(x),vals, dtype=tf.int32)
@tf.function
def get_min_distance(vals):

    smaller = tf.reduce_min(vals[0:2], keepdims=True)
    bigger = tf.reduce_max(vals[0:2], keepdims=True)
    mod = vals[2]

    diff = bigger - smaller # zero if our prediction is correct
    diff_ring = tf.math.abs(mod - bigger + smaller) # maybe abs is no longer necessary
    min_diff = tf.reduce_min(tf.stack([diff, diff_ring]))
    return min_diff

class Meandiff_loss:

    def __init__(self, apply_sum=True, apply_average=True):
        self.__name__ = 'meandiff_loss'
        self.apply_sum=apply_sum
        self.apply_average = apply_average

    def __call__(self, y_true, y_pred, **kwargs):
        return  1 - 1/meandiff_loss_(y_true, y_pred, apply_sum=self.apply_sum, apply_average=self.apply_average) # this should yield a loss between 1 and 0.0001


def meandiff_loss_( y_true, y_pred, apply_sum=True, apply_average=True, as_loss=False):

    """
    Average over the batches
    the sum of the absolute difference between two arrays
    y_true and y_pred are one-hot vectors with the following shape
    batchsize * timesteps * phase classes
    e.g.: 4 * 36 * 5
    First for gt and pred:
    - get the timesteps per phase with the highest probability
    - get the absolute difference between gt and pred
    (- later we can slice each patient by the max value in the corresponding gt indices)
    - sum the diff per entity
    - calc the mean over all examples

    Parameters
    ----------
    y_true :
    y_pred :

    Returns tf.float32 scalar
    -------

    """
    # split gt mask and onehot
    # b, 2, t, phases
    y_true, y_len_msk = tf.unstack(y_true,2,axis=1)
    y_pred, _ = tf.unstack(y_pred,2,axis=1)
    # convert to tensor
    y_true = tf.cast(tf.convert_to_tensor(y_true), tf.float32)
    y_pred = tf.cast(tf.convert_to_tensor(y_pred), tf.float32)
    y_len_msk = tf.cast(tf.convert_to_tensor(y_len_msk), tf.float32)
    # multiply with mask,
    # we are interested in the time step per phase within the gt length
    # b, 36, 5
    #temp_pred = tf.boolean_mask(y_pred, y_len_msk)
    #temp_gt = tf.boolean_mask(y_true,y_len_msk)
    temp_pred = y_pred * y_len_msk
    temp_gt = y_true * y_len_msk
    # make sure axis 1 sums up to one
    #temp_pred = tf.keras.activations.softmax(temp_pred, axis=1)
    #temp_gt = tf.keras.activations.softmax(temp_gt, axis=1)
    #temp_pred = y_pred
    #temp_gt = y_true
    # get the original lengths of each mask in the current batch
    # b, 1
    y_len = tf.cast(tf.reduce_sum(y_len_msk[:,:,0], axis=1),dtype=tf.float32)#
    ############################################ naive test, this works in eager, but not in the loss (line: tf.tile...)
    """@tf.function
    def helper_max(temp):
        # b,36, 5
        max_ = tf.reduce_max(temp, axis=1, keepdims=True)
        # the max value per phase
        # b,1,5, we are interested in b,5 each value in "5" points to the timestep where this phase occures
        # get a mask which points to the max value along axis 1 (for all 36 timesteps)
        cond = tf.cast((temp == max_), tf.float32)
        # b,36,5
        pos = tf.cast(tf.range(36), tf.float32)
        pos = tf.expand_dims(tf.expand_dims(pos, axis=0), axis=-1)
        pos = tf.tile(pos, (temp.shape[0], 1, temp.shape[-1]))
        soft_pos = tf.reduce_sum(cond * pos, axis=1)
        return soft_pos
    gt_idx = helper_max(temp_gt)
    pred_idx = helper_max(temp_pred)"""
    ################################################
    def softargmax(x, axis=1, beta=1e10):
        x = tf.convert_to_tensor(x)
        x_range = tf.range(5, dtype=tf.float32)
        return tf.reduce_sum(tf.nn.softmax(x * beta) * x_range, axis=axis)

    gt_idx = tf.cast(softargmax(temp_gt, axis=1), dtype=tf.float32)
    pred_idx = tf.cast(softargmax(temp_pred, axis=1), dtype=tf.float32)

    #gt_idx = tf.cast(DifferentiableArgmax(temp_gt, axis=1), dtype=tf.float32)#
    #pred_idx = tf.cast(DifferentiableArgmax(temp_pred, axis=1), dtype=tf.float32)
    filled_length = tf.repeat(tf.expand_dims(y_len,axis=1),5,axis=1)
    #print('gt_idx shape: {}'.format(gt_idx.shape))
    #print('pred_idx shape: {}'.format(pred_idx.shape))
    #print('filled shape: {}'.format(filled_length.shape))
    # b, 5, 3
    stacked = tf.stack([gt_idx, pred_idx, filled_length], axis=-1)
    # sum the error per entity, and calc the mean over the batches
    # for each entity in batch ==> 5, 3 in stacked
    diffs = tf.map_fn(lambda x: get_min_dist_for_list_loss(x), stacked, dtype=tf.float32)
    if apply_sum: diffs = tf.cast(tf.reduce_sum(diffs, axis=1),tf.float32)
    if apply_average: diffs = tf.reduce_mean(diffs)
    #tf.math.greater_equal(diffs, 0.), 'distance cant be negative'
    return diffs

@tf.function
def get_min_dist_for_list_loss(vals):
    # vals has the shape 5, 3
    # for each phase tuple (gt,pred,length)
    return tf.map_fn(lambda x :get_min_distance_loss(x),vals, dtype=tf.float32)
@tf.function
def get_min_distance_loss(vals):
    smaller = tf.reduce_min(vals[0:2], keepdims=True)
    bigger = tf.reduce_max(vals[0:2], keepdims=True)
    mod = vals[2]
    diff = bigger - smaller
    diff_ring = tf.math.abs(mod - bigger + smaller)# we need to use the abs to avoid 0 - 0
    min_diff = tf.reduce_min(tf.stack([diff, diff_ring]))
    #tf.math.greater_equal(min_diff, 0.)
    return min_diff

import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras.layers import Layer, Lambda


@tf.function
def DifferentiableArgmax(inputs, axis=-1):
    # if it doesnt sum to one: normalize
    @tf.function
    def prob2oneHot(x):
        # len should be slightly larger than the length of x
        len = 40
        a = tf.math.pow(len * x, 10)
        sum_a = tf.reduce_sum(a, axis=axis)
        sum_a = tf.expand_dims(sum_a, axis=axis)
        onehot = tf.divide(a, sum_a)

        return onehot

    onehot = prob2oneHot(inputs)
    onehot = prob2oneHot(onehot)
    onehot = prob2oneHot(onehot)

    @tf.function
    def onehot2token(x):
        cumsum = tf.cumsum(x, axis=axis, exclusive=True, reverse=True)
        rounding = 2 * (tf.clip_by_value(cumsum, clip_value_min=.5, clip_value_max=1) - .5)
        token = tf.reduce_sum(rounding, axis=axis)
        return token

    token = onehot2token(onehot)
    return token

class CCE_combined(tf.keras.losses.Loss):

    def __init__(self, masked=True, smooth=0.8, transposed=True):

        super().__init__(name='mse_cce_t')
        self.masked = masked
        self.smooth = smooth
        self.transposed = transposed
        self.mse = MSE(masked=False)
        self.cce = CCE(masked=masked, smooth=smooth, transposed=transposed)
        self.cce_weight = 0.5
        self.mse_weight = 0.5

    def __call__(self, y_true, y_pred, **kwargs):

        return self.cce_weight*self.cce(y_true, y_pred) + self.mse_weight * self.mse(y_true,y_pred)


class CCE(tf.keras.losses.Loss):

    def __init__(self, masked=False, smooth=0, transposed=False):

        super().__init__(name='cce_{}_{}_{}'.format(masked,smooth,transposed))
        self.masked = masked
        self.smooth = smooth
        self.transposed = transposed

    def __call__(self, y_true, y_pred, **kwargs):

        if y_true.shape[1] == 2: # this is a stacked onehot vector
            y_true, y_msk = tf.unstack(y_true, num=2, axis=1)
            y_pred, _ = tf.unstack(y_pred, num=2, axis=1)

            if self.masked:
                y_msk = tf.cast(y_msk, tf.float32)
                y_true = y_true * y_msk
                y_pred = y_pred * y_msk

        if self.transposed:
            #, perm=[0, 2, 1]
            y_true = tf.transpose(y_true, perm=[0,2,1,3]),
            y_pred = tf.transpose(y_pred, perm=[0,2,1,3])

        '''y_true = tf.nn.softmax(y_true, axis=1)
        y_pred = tf.nn.softmax(y_pred, axis=1)'''

        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=self.smooth)(y_true, y_pred)
        if self.transposed:
            #loss = tf.transpose(loss, perm=[0,2,1])
            pass

        return tf.reduce_mean(loss)



class MSE:

    def __init__(self, masked=False, loss_fn=tf.keras.losses.mse, onehot=False):

        #super().__init__(name='MSE_{}'.format(masked))
        self.__name__ = 'MSE_{}'.format(masked)
        self.masked = masked
        self.loss_fn = loss_fn
        self.onehot = onehot



    def __call__(self, y_true, y_pred, **kwargs):



        if self.masked:
            if y_pred.shape[1] == 2:  # this is a stacked onehot vector
                y_true, y_msk = tf.unstack(y_true, num=2, axis=1)
                y_pred, zeros = tf.unstack(y_pred, num=2, axis=1)
                # masking works only if we have the gt stacked
                y_msk = tf.cast(y_msk, tf.float32)  # weight the true cardiac cycle by zero and one
                y_true = y_msk * y_true
                y_pred = y_msk * y_pred
        elif self.onehot:
            zeros = tf.zeros_like(y_true[:, 0], tf.float32)
            ones = tf.ones_like(y_true[:, 1], tf.float32)
            msk = tf.stack([ones, zeros], axis=1)
            y_true, y_pred =  msk * y_true, msk * y_pred
            # b, 2,

        if self.loss_fn == 'cce':
            # recent tf version does not support cce with another axis than -1
            # updating tf will break the recent model graph plotting
            y_true, y_pred = tf.transpose(y_true,perm=[0,1,3,2]), tf.transpose(y_pred,perm=[0,1,3,2])
            loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.4,reduction=tf.keras.losses.Reduction.NONE)(y_true,y_pred)
            return loss
        return self.loss_fn(y_true, y_pred)

class SSIM:

    def __init__(self):

        #super().__init__(name='MSE_{}'.format(masked))
        self.__name__ = 'SSIM'



    def __call__(self, y_true, y_pred, **kwargs):

            def get_shape(tensor):
                static_shape = tensor.shape.as_list()
                dynamic_shape = tf.unstack(tf.shape(tensor))
                dims = [s[1] if s[0] is None else s[0]
                        for s in zip(static_shape, dynamic_shape)]
                return dims

            shape_ytrue = get_shape(y_true)
            t_shape = (shape_ytrue[0],shape_ytrue[-3],shape_ytrue[-2],shape_ytrue[1]*shape_ytrue[2])
            #from skimage.metrics import structural_similarity as ssim_fn
            #ssim = ssim_fn(im1=y_true, im2=y_pred, multichannel=True)
            img1 = tf.reshape(tensor=y_true, shape=t_shape)
            img2 = tf.reshape(tensor=y_pred, shape=t_shape)
            ssim = tf.image.ssim(img1, img2, max_val=1.0, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
            return 1- ssim


def mse_wrapper(y_true,y_pred):
    y_true, y_len = tf.unstack(y_true,num=2, axis=1)
    y_pred, _ = tf.unstack(y_pred,num=2, axis=1)

    return tf.keras.losses.mse(y_true, y_pred)

def ca_wrapper(y_true, y_pred):
    y_true, y_len = tf.unstack(y_true,num=2, axis=1)
    y_pred, _ = tf.unstack(y_pred,num=2, axis=1)
    return tf.keras.metrics.categorical_accuracy(y_true, y_pred)

class Grad:
    """
    N-D gradient loss.
    loss_mult can be used to scale the loss value - this is recommended if
    the gradient is computed on a downsampled vector field (where loss_mult
    is equal to the downsample factor).
    """

    def __init__(self, penalty='l1', loss_mult=None, vox_weight=None):
        self.penalty = penalty
        self.loss_mult = loss_mult
        self.vox_weight = vox_weight

    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            yp = K.permute_dimensions(y, r)
            dfi = yp[1:, ...] - yp[:-1, ...]

            if self.vox_weight is not None:
                w = K.permute_dimensions(self.vox_weight, r)
                # TODO: Need to add square root, since for non-0/1 weights this is bad.
                dfi = w[1:, ...] * dfi

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)

        return df

    def loss(self, _, y_pred):
        """
        returns Tensor of size [bs]
        """
        #y_pred = tf.where(tf.math.is_nan(y_pred), tf.zeros_like(y_pred), y_pred)
        #norm = tf.norm(y_pred, axis=-1)
        #return norm /

        if self.penalty == 'l1':
            dif = [tf.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        #y_pred = tf.transpose(y_pred, [0, 1, 2, 4, 3])
        #print(y_pred.shape)
        #norm = tf.norm(y_pred, ord='euclidean', axis=-1)
        #dif = dif + norm

        df = [tf.reduce_mean(K.batch_flatten(f), axis=-1) for f in dif]
        grad = tf.add_n(df) / len(df)
        # ideally this should penalize unnecessary deformation in black areas
        #temp = tf.where(tf.math.is_nan(y_pred), tf.zeros_like(y_pred), y_pred)
        #norm = tf.norm(temp, ord='euclidean', axis=-1)
        #norm = tf.reduce_mean(norm)

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None, eps=1e-5):
        self.win = win
        self.eps = eps

    def ncc(self, I, J):
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(I.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if self.win is None:
            self.win = [9] * ndims

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        # compute filters
        in_ch = J.get_shape().as_list()[-1]
        sum_filt = tf.ones([*self.win, in_ch, 1])
        strides = 1
        if ndims > 1:
            strides = [1] * (ndims + 2)

        # compute local sums via convolution
        padding = 'SAME'
        I_sum = conv_fn(I, sum_filt, strides, padding)
        J_sum = conv_fn(J, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win) * in_ch
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size  # TODO: simplify this
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return mean cc for each entry in batch
        return tf.reduce_mean(K.batch_flatten(cc), axis=-1)

    def loss(self, y_true, y_pred):
        return - self.ncc(y_true, y_pred)

class MSE_:
    """
    Sigma-weighted mean squared error for image reconstruction.
    """

    def __init__(self, image_sigma=1.0):
        self.image_sigma = image_sigma

    def loss(self, y_true, y_pred):
        return 1.0 / (self.image_sigma ** 2) * K.mean(K.square(y_true - y_pred))


def dice_coef_loss(y_true, y_pred):
    # converges from 1 to 0
    smooth = 1.

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1-(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)