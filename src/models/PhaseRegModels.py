import logging
from src.models.KerasLayers import ConvEncoder, get_angle_tf, get_idxs_tf, get_centers_tf, ComposeTransform, \
    conv_layer_fn, ConvBlock

import sys
import numpy as np
import tensorflow
import tensorflow as tf
from tensorflow import keras
import keras.layers as KL
from keras.layers import Input
from keras.models import Model
from tensorflow.python.keras import metrics as metr
from keras.layers import Dropout, BatchNormalization, TimeDistributed
from keras.layers import LSTM, Bidirectional
import math
import atexit

from src.models.Unets import create_unet
from src.utils import Metrics as own_metr

from src.models.ModelUtils import get_optimizer

sys.path.append('src/ext/neuron')
sys.path.append('src/ext/pynd-lib')
sys.path.append('src/ext/pytools-lib')
import src.ext.neuron.neuron.layers as nrn_layers


class PhaseRegressionModel():


    def __init__(self,config, networkname='PhaseRegressionModel'):
        if tf.distribute.has_strategy():
            self.strategy = tf.distribute.get_strategy()
        else:
            # distribute the training with the "mirrored data"-paradigm across multiple gpus if available, if not use gpu 0
            self.strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))
        with self.strategy.scope():

            """from tensorflow.keras import mixed_precision
            policy = mixed_precision.experimental.Policy('mixed_float16')
            mixed_precision.experimental.set_policy(policy)"""
            self.networkname = networkname
            self.input_shape = config.get('DIM', [10, 224, 224])
            self.T_SHAPE = config.get('T_SHAPE', 40)
            self.PHASES = config.get('PHASES', 5)
            self.input_tensor = Input(shape=(self.T_SHAPE, *self.input_shape, 2))
            # define standard values according to the convention over configuration paradigm
            self.activation = config.get('ACTIVATION', 'elu').lower()
            self.kernel_init = config.get('KERNEL_INIT', 'he_normal').lower()
            self.dim = config.get('DIM', [10, 224, 224])
            self.ndims = len(config.get('DIM', [10, 224, 224]))
            self.add_bilstm = config.get('ADD_BILSTM', False)
            self.add_conv_bilstm = config.get('ADD_CONV_BILSTM', False)
            self.lstm_units = config.get('BILSTM_UNITS', 64)
            self.conv_lstm_units = config.get('CONV_BILSTM_UNITS', 64)
            self.add_vect_norm = config.get('ADD_VECTOR_NORM', False)
            self.add_vect_direction = config.get('ADD_VECTOR_DIRECTION', False)
            self.add_flows = config.get('ADD_FLOW', False)
            self.addunetencoding = config.get('ADD_ENC', False)
            self.add_softmax = config.get('ADD_SOFTMAX', False)
            self.softmax_axis = config.get('SOFTMAX_AXIS', 1)
            self.image_loss_weight = config.get('IMAGE_LOSS_WEIGHT', 20)
            self.phase_loss_weight = config.get('PHASE_LOSS_WEIGHT', 1)
            self.flow_loss_weight = config.get('FLOW_LOSS_WEIGHT', 0.01)
            self.final_activation = config.get('FINAL_ACTIVATION', 'relu').lower()
            self.loss = config.get('LOSS', 'mse').lower()
            self.lr = config.get('LEARNING_RATE', 0.001)
            self.mask_loss = config.get('MASK_LOSS', False)
            self.downsample_flow_features = config.get('PRE_GAP_CONV', False)
            self.split_corners = config.get('SPLIT_CORNERS', False)
            self.interp_method = 'linear'
            self.indexing = 'ij'
            # TODO: this parameter is also used by the generator to define the number of channels
            # here we stack the volume within the model
            self.temp_config = config.copy()  # dont change the original config
            self.temp_config['IMG_CHANNELS'] = 2  # we roll the temporal axis and stack t-1, t and t+1 along the last axis
            self.temporal_axis = 1
            self.config = config
            ############################# definition of the layers and blocks ######################################
            # start with very small deformation
            self.Conv = getattr(KL, 'Conv{}D'.format(self.ndims))
            self.Conv_layer = self.Conv(self.ndims, kernel_size=3, padding='same',
                              kernel_initializer=tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=1e-5),
                              name='unet2flow')
            # use a standard U-net, without the final layer for feature extraction (pre-flow)
            self.unet = create_unet(self.temp_config, single_model=False, networkname='3D-Unet')
            # this is a wrapper to re-use the u-net encoding
            self.enc = keras.Model(inputs=self.unet.inputs,outputs=[self.unet.layers[(len(self.unet.layers)//2)-1].output])
            self.st_layer = nrn_layers.SpatialTransformer(interp_method=self.interp_method, indexing=self.indexing, ident=True,
                                                     name='deformable_layer')
            self.st_lambda_layer = keras.layers.Lambda(
                lambda x: self.st_layer([x[..., 0:1], x[..., 1:]]), name='deformable_lambda_layer')

            self.gap = tensorflow.keras.layers.GlobalAveragePooling3D(name='GAP_3D_Layer')
            # concat the current frame with the previous on the last channel
            self.roll_concat_lambda_layer = keras.layers.Lambda(lambda x:
                                                              keras.layers.Concatenate(axis=-1, name='stack_with_moved')(
                                                                  [x,
                                                                   tf.roll(x, shift=-1, axis=self.temporal_axis)]))

            self.norm_lambda = keras.layers.Lambda(
                lambda x: tf.norm(x, ord='euclidean', axis=-1, keepdims=True, name='flow2norm'), name='flow2norm')

            # calculate the direction between the displacement field and a grid with vectors pointing to the center
            # get a tensor with vectors pointing to the center
            # get idxs of one 3D
            # get a tensor with the same shape as the displacement field with vectors toward the center
            # calculate the difference, which should yield a 3D tensor with vectors pointing to the center
            # tile/repeat this v_center vol along the temporal and batch axis
            # calculate the angle of each voxel between the tiled v_center tensor and the displacement tensor
            # concat this tensor as additional feature to the last axis of flow_features
            idx = get_idxs_tf(self.dim)
            c = get_centers_tf(self.dim)
            #print('centers: ',c.dtype)
            centers = c - idx
            centers_tensor = centers[tf.newaxis, ...]
            self.flow2direction_lambda = keras.layers.Lambda(
                lambda x: get_angle_tf(x, centers_tensor), name='flow2direction')

            self.stack_lambda_tf = keras.layers.Lambda(lambda x:
                                                      tf.stack([x,
                                                                tf.zeros_like(x, name='zero_padding')],
                                                               axis=1, name='extend_onehot_by_zeros'),
                                                  name='onehot_lambda')

            forward_conv_lstm_layer = keras.layers.ConvLSTM2D(filters=self.conv_lstm_units,
                                                                 kernel_size=3,
                                                                 strides=1,
                                                                 padding='valid',
                                                                 return_sequences=True,
                                                                 dropout=0.5,
                                                                 name='forward_conv_LSTM')
            backward_conv_lstm_layer = keras.layers.ConvLSTM2D(filters=self.conv_lstm_units,
                                                                  kernel_size=3,
                                                                  strides=1,
                                                                  padding='valid',
                                                                  return_sequences=True,
                                                                  go_backwards=True,
                                                                  dropout=0.5,
                                                                  name='backward_conv_LSTM')
            self.bi_conv_lstm_layer = Bidirectional(forward_conv_lstm_layer, backward_layer=backward_conv_lstm_layer)

            forward_layer = LSTM(self.lstm_units, return_sequences=True, dropout=0.0, name='forward_LSTM')
            backward_layer = LSTM(self.lstm_units, return_sequences=True, dropout=0.0, go_backwards=True,name='backward_LSTM')
            self.bi_lstm_layer = Bidirectional(forward_layer, backward_layer=backward_layer, merge_mode='ave', name='biLSTM')

            forward_layer1 = LSTM(self.lstm_units, return_sequences=True, dropout=0.0, name='forward_LSTM1')
            backward_layer1 = LSTM(self.lstm_units, return_sequences=True, dropout=0.0, go_backwards=True, name='backward_LSTM1')
            self.bi_lstm_layer1 = Bidirectional(forward_layer1, backward_layer=backward_layer1, merge_mode='ave', name='biLSTM1')

            # How to downscale the in-plane/spatial resolution?
            # 1st idea: apply conv layers with a stride
            # b, t, 16, 64, 64, 3/4
            # conv with: n times 4,4,4 filters, valid/no border padding and a stride of 4
            # b, t, 1, 1, 1, n
            # 2nd idea: GAP with/without pre-conv layer which extracts motion features into the channels
            # 3rd idea use the tft.pca module to transform the downstream.
            # This would reduce the dimension of input vectors to output_dim in a way that retains the maximal variance

            downsamples = []
            d_rate = 0.2
            filters_ = 16
            #  b, t, 4, 16, 16, n
            # two times conv with: n times 4,4,4 filters, valid/no border padding and a stride of 4
            # b, t, 1, 4, 4, n
            # conv with: n times 4,4,4 filters, valid/no border padding and a stride of 4
            # how often can we downsample the inplane/spatial resolution until we reach 1
            # n = ln(1/x)/ln0,5

            n = int(math.log(1 / self.dim[-1]) / math.log(0.5))
            z = int(math.log(1 / self.dim[0]) / math.log(0.5))
            for i in range(n):
                if i < z:
                    # downsamples.append(Dropout(d_rate))
                    downsamples.append(
                        # Deformable conv
                        self.Conv(filters=filters_, kernel_size=3, padding='same', strides=1,
                             kernel_initializer=self.kernel_init,
                             activation=self.activation,
                             name='downsample_{}'.format(i)))
                    downsamples.append(keras.layers.MaxPool3D(pool_size=2, padding='same'))
                    filters_ = filters_ * 2
                else:  # stop to down-sample the spatial resolution, continue with 2D conv
                    downsamples.append(
                        keras.layers.Conv2D(filters=filters_, kernel_size=(3, 3), padding='same', strides=1,
                                               kernel_initializer=self.kernel_init,
                                               activation=self.activation,
                                               name='downsample_{}'.format(i))
                    )
                    downsamples.append(keras.layers.MaxPool3D(pool_size=(1, 2, 2), padding='same'))
                downsamples.append(BatchNormalization(axis=-1))

            downsamples = downsamples[:-1]  # remove last BN layer

            self.downsample = keras.Sequential(layers=downsamples, name='downsample_inplane_and_spatial')
            self.final_onehot_conv = keras.layers.Conv1D(filters=self.PHASES, kernel_size=1, strides=1, padding='same',
                                                       kernel_initializer=self.kernel_init, activation=self.final_activation,
                                                       name='pre_onehot')

            ##################################### Layer definition end ##############################################


    def get_model(self):

        with self.strategy.scope():
            print('Shape Input Tensor: {}'.format(self.input_tensor.shape))
            #inputs_spatial_stacked = self.roll_concat_lambda_layer(self.input_tensor)
            # If we roll and stack the input here,
            # we would ignore that the volumes are repeated and stop not at cardiac cycle
            # This would force the model to learn the motion from the middle of a cardiac cycle to the first timestep
            # Additionally this will result in different model input/target for the border cases 0 and t
            #inputs_spatial_stacked = keras.layers.Concatenate(axis=1)([inputs_spatial_stacked[:,:-1], inputs_spatial_stacked[:,-2:-1]])

            pre_flows = TimeDistributed(self.unet, name='4d-p2p-unet')(self.input_tensor)
            print('Unet output shape: {}'.format(pre_flows.shape))
            flows = TimeDistributed(self.Conv_layer, name='4d-p2p-flow')(pre_flows)
            print('Flowfield shape: {}'.format(flows.shape))
            transformed = TimeDistributed(self.st_lambda_layer, name='4d-p2p-st')(keras.layers.Concatenate(axis=-1)([self.input_tensor[...,0:1], flows]))
            print('Transformed shape : {}'.format(transformed.shape))
            features_given = False

            if (self.add_vect_norm and self.add_flows): # use the magnitude and flow
                tensor_magnitude = TimeDistributed(self.norm_lambda)(flows)
                flow_features = keras.layers.Concatenate(axis=-1)([flows, tensor_magnitude])
                features_given = True
                print('Inkl flow and norm shape: {}'.format(flow_features.shape))
            elif self.add_vect_norm: # use only the magnitude
                tensor_magnitude = TimeDistributed(self.norm_lambda)(flows)
                flow_features = tensor_magnitude
                features_given = True
                print('Inkl norm shape: {}'.format(flow_features.shape))
            elif self.add_flows: # use only the flow
                flow_features = flows
                features_given = True
                print('Inkl flow shape: {}'.format(flow_features.shape))

            if self.add_vect_direction:
                directions = TimeDistributed(self.flow2direction_lambda)(flows)
                if features_given:
                    flow_features = keras.layers.Concatenate(axis=-1)(
                    [flow_features, directions])  # encode the spatial location of each vector
                else:
                    flow_features = directions
                    features_given = True
                print('flow features inkl directions shape: {}'.format(flow_features.shape))

            # Apply an Bidirectional convLstm layer before downsampling
            # transpose t and z
            if self.add_conv_bilstm:
                flow_features = tf.transpose(flow_features, perm=[0, 2, 1, 3, 4, 5])
                print('transposed: {}'.format(flow_features.shape))
                flow_features = TimeDistributed(self.bi_conv_lstm_layer)(flow_features)
                flow_features = tf.transpose(flow_features, perm=[0, 2, 1, 3, 4, 5])
                print('flow features after Conv2D-LSTM: {}'.format(flow_features.shape))

            if self.downsample_flow_features:
                flow_features = TimeDistributed(self.downsample)(flow_features)
            else:  # use a gap3D layer
                # slice the 3D sequence of features into on sequence per corner
                # as we could align the CMR spatially
                # each 3D-sliced-corner-sequence represents one specific "part" of the heart
                # finally concat them as channel, which makes them available to the LSTM layer
                if self.split_corners: # average per corner per 3D volume, e.g.: top left, top right ...
                    y, x = flow_features.shape[-3:-1]
                    dir_1 = TimeDistributed(self.gap)(flow_features[..., :y // 2, :x // 2, :])
                    dir_2 = TimeDistributed(self.gap)(flow_features[..., y // 2:, :x // 2, :])
                    dir_3 = TimeDistributed(self.gap)(flow_features[..., y // 2:, x // 2:, :])
                    dir_4 = TimeDistributed(self.gap)(flow_features[..., :y // 2, x // 2:, :])
                    flow_features = keras.layers.Concatenate(axis=-1, name='split_corners')([dir_1, dir_2, dir_3, dir_4])
                else: # average per 3D volume
                    flow_features = TimeDistributed(self.gap)(flow_features)
            flow_features = keras.layers.Reshape(target_shape=(flow_features.shape[1], flow_features.shape[-1]))(
                flow_features)
            print('flow features after downsample/gap layer: {}'.format(flow_features.shape))
            # down-sample the flow in-plane
            # Build an encoder with n times conv+relu+maxpool+bn-blocks
            if self.addunetencoding:
                flow_features2 = TimeDistributed(self.enc)(self.input_tensor)
                print('flow features from encoder: {}'.format(flow_features2.shape))
                flow_features2 = TimeDistributed(keras.layers.Conv2D(16, 2, 1, padding='valid'))(flow_features2)
                print('flow features from encoder: {}'.format(flow_features2.shape))
                flow_features2 = TimeDistributed(self.gap)(flow_features2)
                print('flow features from encoder: {}'.format(flow_features2.shape))
                if features_given:
                    flow_features = tf.concat([flow_features, flow_features2], axis=-1)
                else:
                    flow_features = flow_features2

            if self.add_bilstm:
                # min/max normalisation as lambda
                minmax_lambda_tf = keras.layers.Lambda(lambda x:
                                                       (x - tf.reduce_min(x)) / (
                            tf.reduce_max(x) - tf.reduce_min(x) + keras.backend.epsilon()),
                                                       name='minmaxscaling')

                print('Shape before LSTM layers: {}'.format(flow_features.shape))

                flow_features = minmax_lambda_tf(flow_features)
                flow_features = self.bi_lstm_layer(flow_features)
                flow_features = self.bi_lstm_layer1(flow_features)
                print('Shape after LSTM layers: {}'.format(flow_features.shape))

            # input (t,encoding) output (t,5)
            # Dense and conv layers instead of the LSTM layer both overfit more
            # onehot = keras.layers.Dense(units=5, activation=final_activation, kernel_initializer=kernel_init)(flow_features)
            #flow_features = keras.layers.Conv1D(filters=32, kernel_size=3,strides=1, padding='same', kernel_initializer=kernel_init, activation='relu')(flow_features)
            onehot = self.final_onehot_conv(flow_features)
            print('Shape after final conv layer: {}'.format(onehot.shape))
            # add empty tensor with one-hot shape to align with gt
            if self.add_softmax: onehot = keras.activations.softmax(onehot, axis=self.softmax_axis+1)

            # define the model output names
            onehot = self.stack_lambda_tf(onehot)
            onehot = keras.layers.Activation('linear', dtype='float32', name='onehot')(onehot)
            transformed = keras.layers.Activation('linear', name='transformed', dtype='float32')(transformed)
            flows = keras.layers.Activation('linear', name='flows', dtype='float32')(flows)

            outputs = [onehot, transformed, flows]
            from keras.losses import mse
            from src.utils.Metrics import Grad

            weights = {
                'onehot': self.phase_loss_weight,
                'transformed': self.image_loss_weight,
                'flows': self.flow_loss_weight}

            if self.loss == 'cce':
                losses = {
                    'onehot': own_metr.MSE(masked=self.mask_loss, loss_fn='cce', onehot=True),
                    'transformed': own_metr.MSE(masked=self.mask_loss, loss_fn=keras.losses.mse, onehot=False),
                    'flows': Grad('l2').loss}

            elif self.loss == 'ssim':
                losses = {
                    'onehot': own_metr.MSE(masked=self.mask_loss, loss_fn=keras.losses.mse, onehot=True),
                    'transformed': own_metr.SSIM(),
                    'flows': Grad('l2').loss}
            elif self.loss == 'mae':
                losses = {
                    'onehot': own_metr.MSE(masked=self.mask_loss, loss_fn=keras.losses.mse, onehot=True),
                    'transformed': own_metr.MSE(masked=self.mask_loss, loss_fn=keras.losses.mae, onehot=False),
                    'flows': Grad('l2').loss}

            else:  # default fallback --> MSE - works well
                losses = {
                    'onehot': own_metr.MSE(masked=self.mask_loss, loss_fn=keras.losses.mse,onehot=True),
                    'transformed': own_metr.MSE(masked=self.mask_loss, loss_fn=keras.losses.mse,onehot=False),
                    'flows': Grad('l2').loss}


            print('added loss: {}'.format(self.loss))
            model = Model(inputs=[self.input_tensor], outputs=outputs, name=self.networkname)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                loss=losses,
                loss_weights=weights,
                metrics={
                    'onehot': own_metr.Meandiff(),
                    #'onehot': [own_metr.Meandiff(), own_metr.meandiff_loss_]
                    #'transformed': own_metr.SSIM(),
                    # 'flows': Grad('l2').loss
                }
            )
            """[print(i.shape, i.dtype) for i in model.inputs]
            [print(o.shape, o.dtype) for o in model.outputs]
            [print(l.name, l.input_shape, l.dtype) for l in model.layers]"""
            return model

def get_idxs_tf(x):
    return tf.cast(
        tf.reshape(tf.where(tf.ones((x[0], x[1], x[2]))), (x[0], x[1], x[2], 3)),
        tf.float32)


# ST to apply m to an volume
def create_affine_transformer_fixed(config, networkname='affine_transformer_fixed', fill_value=0,
                                    interp_method='linear'):
    """
    Apply a learned transformation matrix to an input image, no training possible
    :param config:  Key value pairs for image size and other network parameters
    :param networkname: string, name of this model scope
    :param fill_value:
    :return: compiled keras model
    """
    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        # distribute the training with the mirrored data paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))

    with strategy.scope():

        inputs = Input((*config.get('DIM', [10, 224, 224]), 1))
        input_displacement = Input((*config.get('DIM', [10, 224, 224]), 3))
        indexing = config.get('INDEXING', 'ij')

        # warp the source with the flow
        y = nrn_layers.SpatialTransformer(interp_method=interp_method, indexing=indexing, ident=False,
                                          fill_value=fill_value)([inputs, input_displacement])

        model = Model(inputs=[inputs, input_displacement], outputs=[y, input_displacement], name=networkname)

        return model


def create_dense_compose(config, networkname='dense_compose_displacement'):
    """
    Compose a single transform from a series of transforms.
    Supports both dense and affine transforms, and returns a dense transform unless all
    inputs are affine. The list of transforms to compose should be in the order in which
    they would be individually applied to an image. For example, given transforms A, B,
    and C, to compose a single transform T, where T(x) = C(B(A(x))), the appropriate
    function call is:
    T = compose([A, B, C])
    :param config:  Key value pairs for image size and other network parameters
    :param networkname: string, name of this model scope
    :param fill_value:
    :return: compiled keras model
    """
    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
    else:
        # distribute the training with the mirrored data paradigm across multiple gpus if available, if not use gpu 0
        strategy = tf.distribute.MirroredStrategy(devices=config.get('GPUS', ["/gpu:0"]))

    with strategy.scope():

        inputs = Input((5, *config.get('DIM', [10, 224, 224]), 3))
        indexing = config.get('INDEXING', 'ij')
        reverse = config.get('REVERSE_COMPOSE', False)
        # warp the source with the flow
        flows = tf.unstack(inputs, axis=1)
        # reverse=True
        # we need to reverse the transforms as we register from t+1 to t.
        # we need to provide the order in which we would apply the compose transforms
        # ED, MS, ES, PF, MD
        # flows:
        # 0= MS->ED, 1=ES->MS, 2=PF->ES, 3=MD->PF, 4=ED->MD
        # e.g. for compose:
        # MS->ED = [0]
        # ES->ED = [1,0]
        # PF->ED = [2,1,0]
        # list(reversed())
        if reverse:
            y = [ComposeTransform(interp_method='linear', shift_center=True, indexing=indexing,
                                  name='Compose_transform{}'.format(i))(list(reversed(flows[:i]))) for i in
                 range(2, len(flows) + 1)]
        else:
            y = [ComposeTransform(interp_method='linear', shift_center=True, indexing=indexing,
                                  name='Compose_transform{}'.format(i))(flows[:i]) for i in range(2, len(flows) + 1)]
        y = tf.stack([flows[0], *y], axis=1)

        model = Model(inputs=[inputs], outputs=[y], name=networkname)

        return model
