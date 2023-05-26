import logging
from logging import info, debug
import sys
import os
import SimpleITK as sitk
from scipy import ndimage
import skimage

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize
from src.data.Dataset import describe_sitk, get_metadata_maybe, copy_meta_and_save, get_patient
import numpy as np
import scipy.ndimage as nd

from albumentations import GridDistortion, RandomRotate90, Compose, ReplayCompose, Flip, Transpose, OneOf, IAAAdditiveGaussianNoise, \
    MotionBlur, MedianBlur, Blur, OpticalDistortion, IAAPiecewiseAffine, CLAHE, IAASharpen, IAAEmboss, \
    RandomBrightnessContrast, HueSaturationValue, ElasticTransform, CenterCrop, PadIfNeeded, RandomBrightness, Downscale, ShiftScaleRotate
import cv2
from src.data.Dataset import copy_meta
from albumentations.augmentations.transforms import PadIfNeeded, GaussNoise, RandomGamma


def load_masked_img(sitk_img_f, mask=False, masking_values = [1,2,3], replace=('img','msk'), mask_labels=[0,1,2,3], maskAll=True, is_mask=False):

    """
    Wrapper for opening a dicom image, this wrapper could also load the corresponding segmentation map and mask the loaded image on the fly
     if mask == True use the replace wildcard to open the corresponding segmentation mask
     Use the values given in mask_labels to transform the one-hot-encoded mask into channel based binary mask
     Mask/cut the CMR image/volume by the given labels in masking_values

    Parameters
    ----------
    sitk_img_f : full filename for a dicom image/volume, could be any format supported by sitk
    mask : bool, if the sitk image loaded should be cropped by any label of the corresponding mask
    masking_values : list of int, defines the area/labels which should be cropped from the original CMR
    replace : tuple of replacement string to get from the image filename to the mask filename
    mask_labels : list of int
    maskAll: bool,  true: mask all timesteps of the CMR by the mask,
                    false: return the raw CMR for timesteps without a mask
    """

    assert os.path.isfile(sitk_img_f), 'no valid image: {}'.format(sitk_img_f)
    img_original = sitk.ReadImage(sitk_img_f, sitk.sitkFloat32)

    if mask:
        sitk_mask_f = sitk_img_f.replace(replace[0], replace[1])
        msk_original = sitk.ReadImage(sitk_mask_f)
        
        img_nda = sitk.GetArrayFromImage(img_original)
        msk_nda = transform_to_binary_mask(sitk.GetArrayFromImage(msk_original), mask_values=mask_labels)
                    
        # mask by different labels, sum up all masked channels
        temp = np.zeros(img_nda.shape)
        if maskAll: # mask all timesteps
            for c in masking_values:
                # mask by different labels, sum up all masked channels
                temp += img_nda * msk_nda[..., c].astype(np.bool)
            sitk_img = sitk.GetImageFromArray(temp)
        else:
            for t in range(img_nda.shape[0]):
                if msk_nda[t].sum() > 0: # mask only timesteps with a given mask
                    for c in masking_values:
                        # mask by different labels, sum up all masked channels
                        temp[t] += img_nda[t] * msk_nda[t][..., c].astype(np.bool)
            sitk_img = sitk.GetImageFromArray(temp)

        # copy metadata
        for tag in img_original.GetMetaDataKeys():
            value = get_metadata_maybe(img_original, tag)
            sitk_img.SetMetaData(tag, value)
        sitk_img.SetSpacing(img_original.GetSpacing())
        sitk_img.SetOrigin(img_original.GetOrigin())
                    
        img_original = sitk_img
                
    return img_original

def load_msk(f_name, valid_labels=None):

    if valid_labels is None:
        valid_labels = [0, 1, 2, 3]
    msk_sitk = sitk.ReadImage(f_name)
    msk_nda = sitk.GetArrayFromImage(msk_sitk)
    msk_binary = np.squeeze(transform_to_binary_mask(msk_nda, mask_values=valid_labels))

    msk_b_sitk = sitk.GetImageFromArray(msk_binary.astype(np.float32))

    # copy metadata
    for tag in msk_sitk.GetMetaDataKeys():
        value = get_metadata_maybe(msk_sitk, tag)
        msk_b_sitk.SetMetaData(tag, value)
    msk_b_sitk.SetSpacing(msk_sitk.GetSpacing())
    msk_b_sitk.SetOrigin(msk_sitk.GetOrigin())
    return msk_b_sitk

def resample_t_of_4d(sitk_img, t_spacing=20, interpolation=sitk.sitkLinear, ismask=False):
    '''

    Parameters
    ----------
    sitk_img :
    t_spacing :
    interpolation :
    ismask :

    Returns
    -------

    '''

    from src.data.Dataset import split_one_4d_sitk_in_list_of_3d_sitk

    debug('before resample: {}'.format(sitk_img.GetSize()))
    debug('before resample: {}'.format(sitk_img.GetSpacing()))
    z_spacing = sitk_img.GetSpacing()[2]
    # unstack 4D volume along t
    images = split_one_4d_sitk_in_list_of_3d_sitk(sitk_img, axis=1)

    temp_slice = images[0]
    t_size = temp_slice.GetSize()
    spacing_old = temp_slice.GetSpacing()

    # calculate the sampling factor, and the corresponding new size in t
    sampling_factor = spacing_old[-1] / t_spacing
    debug('Sampling factor: {}'.format(sampling_factor))
    new_t_size = int(np.floor(t_size[-1] * sampling_factor))
    target_size = (t_size[0], t_size[1], new_t_size)
    debug('2D+t target size: {}'.format(target_size))
    target_spacing = (spacing_old[0], spacing_old[1], t_spacing)

    resampled_cmr = list(
        map(lambda x:
            resample_3D(x, size=target_size, spacing=target_spacing, interpolate=interpolation), images))

    # get array from list of 2d+t sitk images, stack along z, unstack along t --> list of 3D volumes
    resampled_cmr_nda = list(map(sitk.GetArrayFromImage, resampled_cmr))
    debug('Before stack: {}'.format(resampled_cmr_nda[0].shape))
    resampled_cmr_nda = np.stack(resampled_cmr_nda, axis=1)

    if ismask:
        idxs_orig = get_mask_idxs(sitk_img)
        # scale the idx by the resampling factor, use this to filter the doubled resampled masks
        # resampling and filtering of the masks to avoid double time steps/phases
        idxs_scaled = np.round(idxs_orig * sampling_factor).astype(int)
        debug('scaled idx: {}'.format(idxs_scaled))
        idxs_scaled = np.clip(idxs_scaled, a_min=0, a_max=resampled_cmr_nda.shape[0] - 1)
        debug('scaled, clipped idx: {}'.format(idxs_scaled))
        temp = np.zeros_like(resampled_cmr_nda)
        temp[idxs_scaled] = resampled_cmr_nda[idxs_scaled]
        resampled_cmr_nda = temp

        mask_idx = np.max(resampled_cmr_nda, axis=(1, 2, 3))
        debug('max: {}'.format(mask_idx))
        idxs_resampled = np.where(mask_idx >= 3)[0].astype(int)
        debug('after stack: {}'.format(resampled_cmr_nda.shape))
        debug('idxs of resampled 4d vol: {}'.format(idxs_resampled))
        assert len(idxs_resampled) == 5

    # split into t x 3D volumes, which is the format we need for sitk to join a series of 3D volumes
    resampled_cmr_nda = np.split(resampled_cmr_nda, indices_or_sections=resampled_cmr_nda.shape[0], axis=0)
    debug('after split: {}'.format(len(resampled_cmr_nda)))
    resampled_cmr_sitk = list(map(lambda x: sitk.GetImageFromArray(np.squeeze(x)), resampled_cmr_nda))
    debug('after as sitk: {} x {}'.format(len(resampled_cmr_nda), resampled_cmr_sitk[0].GetSize()))
    resampled_cmr_sitk = sitk.JoinSeries(resampled_cmr_sitk)

    full_spacing = (spacing_old[0], spacing_old[1], z_spacing, t_spacing)
    resampled_cmr_sitk = copy_meta_and_save(resampled_cmr_sitk, sitk_img, None, overwrite_spacing=full_spacing)
    debug(resampled_cmr_sitk.GetSize())
    debug(resampled_cmr_sitk.GetSpacing())

    return resampled_cmr_sitk


def get_mask_idxs(sitk_img, slice_threshold=2):
    '''
    get valid temporal indices, with a slice threshold, as we have some slices labelled along all time steps
    Parameters
    ----------
    sitk_img : sitk.Image of a 4D mask

    Returns (list) indices of the temporal axis, where we have a mask volume
    -------

    '''
    # from mask2idx for mask and resampled mask
    #
    timesteps = []
    # get indexes for masked volumes
    # filter 3d volumes with less masked slices than threshold
    for t, nda_3d in enumerate(sitk.GetArrayViewFromImage(sitk_img)):
        if nda_3d.max() > 0:  # 3d volume with masks
            masked_slices = 0
            success_ = False
            for slice_ in nda_3d:  # check how many slices are masked
                if slice_.max() > 0:
                    masked_slices += 1
                if masked_slices > slice_threshold:
                    timesteps.append(t)
                    success_ = True
                    break
            if not success_: logging.debug('filter {}t of volume by masked slices threshold'.format(t))
    idxs_orig = np.array(timesteps).astype(int)
    return idxs_orig


def resample_3D(sitk_img, size=(256, 256, 12), spacing=(1.25, 1.25, 8), interpolate=sitk.sitkNearestNeighbor):
    """
    resamples an 3D sitk image or numpy ndarray to a new size with respect to the giving spacing
    This method expects size and spacing in sitk format: x, y, z
    :param sitk_img: sitk.Image
    :param size: (tuple) of int with the following order x,y,z
    :param spacing: (tuple) of float with the following order x,y,z
    :param interpolate:
    :return: the resampled image in the same datatype as submitted, either sitk.image or numpy.ndarray
    """

    return_sitk = True

    if isinstance(sitk_img, np.ndarray):
        return_sitk = False
        sitk_img = sitk.GetImageFromArray(sitk_img)

    assert (isinstance(sitk_img, sitk.Image)), 'wrong image type: {}'.format(type(sitk_img))

    # minor data type corrections
    size = [int(elem) for elem in size]
    spacing = [float(elem) for elem in spacing]

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolate)
    resampler.SetSize(size)
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetOutputSpacing(spacing)
    resampler.SetOutputOrigin(sitk_img.GetOrigin())

    resampled = resampler.Execute(sitk_img)

    # return the same data type as input datatype
    if return_sitk:
        return resampled
    else:
        return sitk.GetArrayFromImage(resampled)

def random_rotate90_2D_or_3D(img, mask, probabillity=0.8):
    logging.debug('random rotate for: {}'.format(img.shape))
    augmented = {'image': None, 'mask': None}

    if isinstance(img, sitk.Image):
        img = sitk.GetArrayFromImage(img).astype(np.float32)

    if isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask).astype(np.float32)

    # dont print anything if no images nor masks are given
    if img is None and mask is None:
        logging.error('No image data given')
        raise ('No image data given in grid dissortion')

    # replace mask with empty slice if none is given
    if mask is None:
        mask = np.zeros(img.shape)

    # replace image with empty slice if none is given
    if img is None:
        img = np.zeros(mask.shape)

    if img.ndim == 2:

        aug = RandomRotate90(p=probabillity)

        params = aug.get_params()
        image_aug = aug.apply(img, **params)
        mask_aug = aug.apply(mask, interpolation=cv2.INTER_NEAREST, **params)

        # apply shift-scale and rotation augmentation on 2d data
        augmented['image'] = image_aug
        augmented['mask'] = mask_aug

    elif img.ndim == 3:
        # apply shif-scale and rotation on 3d data, apply the same transform to all slices
        images = []
        masks = []

        aug = RandomRotate90(p=probabillity)
        params = aug.get_params()
        for z in range(img.shape[0]):
            images.append(aug.apply(img[z, ...], interpolation=cv2.INTER_LINEAR, factor=1,**params))
            masks.append(aug.apply(mask[z, ...], interpolation=cv2.INTER_NEAREST, factor=1,**params))

        augmented['image'] = np.stack(images, axis=0)
        augmented['mask'] = np.stack(masks, axis=0)

    else:
        logging.error('Unsupported dim: {}, shape: {}'.format(img.ndim, img.shape))
        raise ('Wrong shape Exception in: {}'.format('show_2D_or_3D()'))

    return augmented['image'], augmented['mask']

def augmentation_compose_2d_3d_4d(img, mask, probabillity=1, config=None):
    """
    Apply an compisition of different augmentation steps,
    either on 2D or 3D image/mask pairs,
    apply
    :param img:
    :param mask:
    :param probabillity:
    :return: augmented image, mask
    """
    #logging.debug('random rotate for: {}'.format(img.shape))
    if config is None:
        config = {}
    return_image_and_mask = True
    img_given = True
    mask_given = True


    if isinstance(img, sitk.Image):
        img = sitk.GetArrayFromImage(img).astype(np.float32)

    if isinstance(mask, sitk.Image):
        mask = sitk.GetArrayFromImage(mask).astype(np.float32)

    # dont print anything if no images nor masks are given
    if img is None and mask is None:
        logging.error('No image data given')
        raise ('No image data given in augmentation compose')

    # replace mask with empty slice if none is given
    if mask is None:
        return_image_and_mask = False
        mask_given = False

    # replace image with empty slice if none is given
    if img is None:
        return_image_and_mask = False
        img_given = False

    targets = {}
    data = {}
    img_placeholder = 'image'
    mask_placeholder = 'mask'

    if img.ndim == 2:
        data = {"image": img, "mask": mask}

    if img.ndim == 3:
        middle_z = len(img)//2
        if mask_given:
            m_ = mask[middle_z]
        else:
            m_ = mask
        # take an image, mask pair from the middle part of the volume
        data = {"image": img[middle_z], "mask": m_}

        # add each slice of the image/mask stacks into the data dictionary
        for z in range(img.shape[0]):
            # add the other slices to the data dict
            if img_given: data['{}{}'.format(img_placeholder,z)] = img[z,...]
            if mask_given:data['{}{}'.format(mask_placeholder, z)] = mask[z, ...]
            # define the target group,
            # which slice is a mask and which an image (different interpolation)
            if img_given: targets['{}{}'.format(img_placeholder,z)] = 'image'
            if mask_given: targets['{}{}'.format(mask_placeholder, z)] = 'mask'

    if img.ndim ==4:
        middle_t = img.shape[0] // 2
        middle_z = img.shape[1] // 2
        # take an image, mask pair from the middle part of the volume and time
        if mask_given:
            data = {"image": img[middle_t][middle_z], "mask": m_}
        else:
            data = {"image": img[middle_t][middle_z]}


        for t in range(img.shape[0]):
            # add each slice of the image/mask stacks into the data dictionary
            for z in range(img.shape[1]):
                # add the other slices to the data dict
                if img_given: data['{}_{}_{}'.format(img_placeholder, t, z)] = img[t,z, ...]
                if mask_given:data['{}_{}_{}'.format(mask_placeholder, t, z)] = mask[t,z, ...]
                # define the target group,
                # which slice is a mask and which an image (different interpolation)
                if img_given: targets['{}_{}_{}'.format(img_placeholder, t, z)] = 'image'
                if mask_given: targets['{}_{}{}'.format(mask_placeholder, t,z)] = 'mask'

    # create a callable augmentation composition
    aug = _create_aug_compose(p=probabillity, targets=targets, config=config)

    # apply the augmentation
    augmented = aug(**data)
    logging.debug(augmented['replay'])

    if img.ndim == 3:
        images = []
        masks = []
        for z in range(img.shape[0]):
            # extract the augmented slices in the correct order
            if img_given: images.append(augmented['{}{}'.format(img_placeholder,z)])
            if mask_given:masks.append(augmented['{}{}'.format(mask_placeholder, z)])
        if img_given: augmented['image'] = np.stack(images,axis=0)
        if mask_given: augmented['mask'] = np.stack(masks, axis=0)

    if img.ndim == 4:
        img_4d = []
        mask_4d = []
        for t in range(img.shape[0]):
            images = []
            masks = []
            for z in range(img.shape[1]):
                # extract the augmented slices in the correct order
                if img_given: images.append(augmented['{}_{}_{}'.format(img_placeholder,t,z)])
                if mask_given: masks.append(augmented['{}_{}_{}'.format(mask_placeholder,t, z)])
            if img_given: img_4d.append(np.stack(images,axis=0))
            if mask_given: mask_4d.append(np.stack(masks, axis=0))

        if img_given: augmented['image'] = np.stack(img_4d,axis=0)
        if mask_given: augmented['mask'] = np.stack(mask_4d, axis=0)


    if return_image_and_mask:
        return augmented['image'], augmented['mask']
    else:
        # dont return the fake augmented masks if none where given
        return augmented['image']


def _create_aug_compose(p=1, border_mode=cv2.BORDER_CONSTANT, val=0, targets=None, config=None):
    """
    Create an Albumentations Reply compose augmentation based on the config params
    Parameters
    ----------
    p :
    border_mode :
    val :
    targets :
    config :
    Note for the border mode from openCV:
    BORDER_CONSTANT    = 0,
    BORDER_REPLICATE   = 1,
    BORDER_REFLECT     = 2,
    BORDER_WRAP        = 3,
    BORDER_REFLECT_101 = 4,
    BORDER_TRANSPARENT = 5,
    BORDER_REFLECT101  = BORDER_REFLECT_101,
    BORDER_DEFAULT     = BORDER_REFLECT_101,
    BORDER_ISOLATED    = 16,

    Returns
    -------

    """
    if config is None:
        config = {}
    if targets is None:
        targets = {}
    prob = config.get('AUGMENT_PROB', 0.8)
    border_mode = config.get('BORDER_MODE', border_mode)
    val = config.get('BORDER_VALUE', val)
    augmentations = []
    if config.get('RANDOMROTATE', False):augmentations.append(RandomRotate90(p=0.2))
    if config.get('SHIFTSCALEROTATE', False): augmentations.append(ShiftScaleRotate(p=prob, rotate_limit=0,shift_limit=0.025, scale_limit=0,value=val, border_mode=border_mode))
    if config.get('GRIDDISTORTION', False): augmentations.append(GridDistortion(p=prob, value=val,border_mode=border_mode))
    if config.get('DOWNSCALE', False): augmentations.append(Downscale(scale_min=0.9, scale_max=0.9, p=prob))
    return ReplayCompose(augmentations, p=p,
        additional_targets=targets)

def transform_to_binary_mask(mask_nda, mask_values=[0, 1, 2, 3]):
    """
    Transform from a value-based representation to a binary channel based representation
    :param mask_nda:
    :param mask_values:
    :return:
    """
    # transform the labels to binary channel masks

    mask = np.zeros((*mask_nda.shape, len(mask_values)), dtype=np.bool)
    for ix, mask_value in enumerate(mask_values):
        mask[..., ix] = mask_nda == mask_value
    return mask


def from_channel_to_flat(binary_mask, start_c=0):

    """
    Transform a tensor or numpy nda from a channel-wise (one channel per label) representation
    to a value-based representation
    :param binary_mask:
    :return:
    """
    # convert to bool nda to allow later indexing
    binary_mask = binary_mask >= 0.5

    # reduce the shape by the channels
    temp = np.zeros(binary_mask.shape[:-1], dtype=np.uint8)

    for c in range(binary_mask.shape[-1]):
        temp[binary_mask[..., c]] = c + start_c
    return temp


def clip_quantile(img_nda, upper_quantile=.999, lower_boundary=0):
    """
    clip to values between 0 and .999 quantile
    :param img_nda:
    :param upper_quantile:
    :return:
    """


    ninenine_q = np.quantile(img_nda.flatten(), upper_quantile, overwrite_input=False)

    return np.clip(img_nda, lower_boundary, ninenine_q)


def normalise_image(img_nda, normaliser='minmax'):
    """
    Normalise Images to a given range,
    normaliser string repr for scaler, possible values: 'MinMax', 'Standard' and 'Robust'
    if no normalising method is defined use MinMax normalising
    :param img_nda:
    :param normaliser:
    :return:
    """
    # ignore case
    normaliser = normaliser.lower()

    if normaliser == 'standard':
        return (img_nda - np.mean(img_nda)) / (np.std(img_nda)+ sys.float_info.epsilon)

        #return StandardScaler(copy=False, with_mean=True, with_std=True).fit_transform(img_nda)
    elif normaliser == 'robust':
        return RobustScaler(copy=False, quantile_range=(0.0, 95.0), with_centering=True,
                            with_scaling=True).fit_transform(img_nda)
    else:
        return (img_nda - img_nda.min()) / (img_nda.max() - img_nda.min() + sys.float_info.epsilon)


def pad_and_crop(ndarray, target_shape=(10, 10, 10)):
    """
    Center pad and crop a np.ndarray in one step
    Accepts any shape (2D,3D, ..nD) to a given target shape
    Expects ndarray.ndim == len(target_shape)
    This method is idempotent, which means the pad operation is the numeric inverse of the crop operation
    Pad and crop must be the complementary,
    In cases of non odd shapes in any dimension this method defines the center as:
    pad_along_dimension_n = floor(border_n/2),floor(border_n/2)+1
    crop_along_dimension_n = floor(margin_n/2)+1, floor(margin_n/2)
    Parameters:
    ----------
    ndarray : numpy.ndarray of any shape
    target_shape : must have the same length as ndarray.ndim

    Returns np.ndarray with each axis either pad or crop
    -------

    """
    cropped = np.zeros(target_shape, dtype=np.float32)
    target_shape = np.array(target_shape)
    logging.debug('input shape, crop_and_pad: {}'.format(ndarray.shape))
    logging.debug('target shape, crop_and_pad: {}'.format(target_shape))

    diff = ndarray.shape - target_shape

    # divide into summands to work with odd numbers
    # take the same numbers for left or right padding/cropping if the difference is dividable by 2
    # else take floor(x),floor(x)+1 for PAD (diff<0)
    # else take floor(x)+1, floor(x) for CROP (diff>0)
    # This behaviour is the same for each axis
    d = list((int(x // 2), int(x // 2)) if x % 2 == 0 else (int(np.floor(x / 2)), int(np.floor(x / 2) + 1)) if x<0 else (int(np.floor(x / 2)+1), int(np.floor(x / 2))) for x in diff)
    # replace the second slice parameter if it is None, which slice until end of ndarray
    d = list((abs(x), abs(y)) if y != 0 else (abs(x), None) for x, y in d)
    # create a bool list, negative numbers --> pad, else --> crop
    pad_bool = diff < 0
    crop_bool = diff > 0

    # create one slice obj for cropping and one for padding
    pad = list(i if b else (None, None) for i, b in zip(d, pad_bool))
    crop = list(i if b else (None, None) for i, b in zip(d, crop_bool))

    # Create one tuple of slice calls per pad/crop
    # crop or pad from dif:-dif if second param is not None, else replace by None to slice until the end
    # slice params: slice(start,end,steps)
    pad = tuple(slice(i[0], -i[1]) if i[1] != None else slice(i[0], i[1]) for i in pad)
    crop = tuple(slice(i[0], -i[1]) if i[1] != None else slice(i[0], i[1]) for i in crop)

    # crop and pad in one step
    cropped[pad] = ndarray[crop]
    return cropped


from math import atan2, degrees
def get_angle2x(p1, p2):
    '''
    Calc the angle between two points and the x-axis
    Parameters
    ----------
    p1 : tuple x,y
    p2 : tuple x,y

    Returns angle in degree
    -------

    '''
    angle = 0
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
    if y2 > y1:
        angle = degrees(atan2(y2-y1, x2-x1))
    else:
        angle = degrees(atan2(y1-y2, x1-x2))
    return angle

def get_ip_from_mask_3d(msk_3d, debug=False, keepdim=False, rev=False):
    '''
    Returns two lists of RV insertion points (y,x)-coordinates
    For a standard SAX orientation:
    the first list belongs to the anterior IP and the second to the inferior IP
    Parameters
    ----------
    msk_3d : (np.ndarray) with z,y,x
    debug : (bool) print additional info
    keepdim: (bool) returns two lists of the same length as z, slices where no RV IPs were found are represented by an tuple of None
    rev: (bool), return the coordinates as x,y tuples (for comparison with matrix based indexing)

    Returns tuple of lists with points (y,x)-coordinates
    -------

    '''
    first_ips = []
    second_ips = []
    for msk2d in msk_3d:
        try:
            first, second = get_ip_from_2dmask(msk2d, debug=debug, rev=rev)
            if (first and second) or keepdim:
                first_ips.append(first)
                second_ips.append(second)
        except Exception as e:
            print(str(e))
            pass

    return first_ips, second_ips

def align_inplane_with_ip(model_inputs, msk_file_name, roll2septum=True, roll2lvbood=False, rotate=True, translate=True):
    '''
    Rotate a 4d SAX CMR stack according to the RV insertion points of a corresponding mask
    Returns the same 4d SAX stack but in-plane rotated to
    a 90 degree angle between the mean RV insertion points and the x-axis
    Parameters
    ----------
    model_inputs : (np.ndarray) with t,z,y,x,c
    msk_file_name : (str) full filename to a 4d mask with 0=background, 1=RV, 2=MYO, 3=LV

    Returns the 4d SAX CMR stack in-plane rotated
    -------

    '''
    if roll2lvbood or roll2septum: # for this alignment we need a mask
        if type(msk_file_name) == type(''):
            mask = sitk.GetArrayFromImage(sitk.ReadImage(msk_file_name))
        else:
            mask = msk_file_name
        # t,z,y,x --> numpy order of dimensions
        # Find the first labelled time step, could also be done for all labelled time steps
        if mask.ndim==3:
            mask3d = mask
        elif mask.shape[0] ==1:
            mask3d = mask[0]
        else:
            i = get_first_idx(mask)
            mask3d = mask[i]

        import matplotlib.pyplot as plt
        from src.visualization.Visualize import show_2D_or_3D
        # first roll to center of mean septum,
        # than rotate to a 90 degree angle between the septum and the x-axis
        # Get the anterior and inferior insertion points for all valid slices
        fips, sips = get_ip_from_mask_3d(mask3d)
        # average both points to find the mean fip and sip per volume
        fip = np.array(fips).mean(axis=0)
        sip = np.array(sips).mean(axis=0)


    # Move to center of lv bloodpool
    if roll2lvbood:
        center = nd.center_of_mass(mask3d==3)
        center = center[1:] # ignore z-axis for translation
    elif roll2septum: # roll2septum/mean septum wall is default fallback
        center = np.mean([fip,sip], axis=0).astype(int)
        center = center[::-1]
    else: # use the mean squared error along t as definition of the center of change

        ##### new
        center = get_center_by_mse(model_inputs)

    if translate:
        # translation to center of septum or lv bloodpool
        ny, nx = model_inputs.shape[-2:]
        ry = int(ny//2-center[0])
        rx = int(nx//2-center[1])
        model_inputs = np.roll(model_inputs, ry, axis=-2) # roll the y-axis
        model_inputs = np.roll(model_inputs, rx, axis=-1) # roll the x-axis

    if rotate:
        # Calc the angle between the mean septum and the x-axis
        ip_angle = get_angle2x(fip, sip)
        # How much do we want to rotate
        rot_angle = ip_angle - 90
        # Rotate the 4D volume in-plane (x,y-axis)
        model_inputs = ndimage.rotate(model_inputs, angle=rot_angle, reshape=False, order=1, axes=(-2, -1))

    return model_inputs


def get_center_by_mse(model_inputs):
    """

    Parameters
    ----------
    model_inputs : 4d CMR as numpy ndarray with t,z,y,x

    Returns tuple (y,x)
    -------

    """
    import tensorflow as tf
    from src.visualization.Visualize import show_2D_or_3D
    import matplotlib.pyplot as plt
    nz, ny, nx = model_inputs.shape[-3:]
    border_percentage = 20
    gray_percentile_threshold = .99
    lower_gray_percentile_threshold = .90 # .90
    mse_threshold = 90
    gaus_sigma = 3
    nyx = np.array([int(ny), int(nx)])
    border = ((nyx / 100) * border_percentage).astype(np.int)
    temp = clip_quantile(model_inputs, gray_percentile_threshold,
                         lower_boundary=-1 * np.quantile(-1 * model_inputs, lower_gray_percentile_threshold))
    temp = normalise_image(temp[:-1, nz // 4:, border[0]:-border[0], border[1]:-border[1]], normaliser='standard')
    temp_roll = np.roll(temp, shift=-1, axis=0)
    # ignore the mse between frame t0 and t-1, as this might reflect the cut cmr sequence
    mse_ = tf.keras.metrics.mean_squared_error(temp[..., None], temp_roll[..., None]).numpy()
    # normalise per timestep, less change in diastole
    # mse_ = np.stack([normalise_image(elem, normaliser='minmax') for elem in mse_], axis=0)
    mse_smooth = ndimage.gaussian_filter(mse_, gaus_sigma)
    mse_mean = np.mean(mse_smooth, axis=0)
    from skimage.measure import label
    def getLargestCC(segmentation):
        labels = label(segmentation)
        assert (labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        return largestCC

    mse_mean_mask = mse_mean > np.percentile(mse_mean, mse_threshold)
    #detected_percentage = (100 / mse_mean_mask.size) * mse_mean_mask.sum()
    #if detected_percentage < 10:
    mse_mean_mask_filter = getLargestCC(mse_mean_mask)
    reduced_part = mse_mean_mask_filter.sum()/mse_mean_mask.sum()
    # else:
    #     mse_mean_mask_filter = mse_mean_mask
    # extract one mse center per t, this should be more robust to outliers than com of a mean mse
    try:  # smoothed center, move center from vol center towards the mse center of mass, performs little worse
        center = nd.center_of_mass(mse_mean_mask_filter)
        if reduced_part < 0.7: # this is a strong indicator that there are more than two possible focuses, better use the center
            center = np.array(mse_mean_mask_filter.shape) // 2  # center of volume
            #center = (np.array(center_vol) + np.array(center)) // 2
        center = np.array(center[1:]) + border
        # centers = np.array([nd.center_of_mass(elem[:, border:-border, border:-border]) for elem in mse_mean_mask if elem[:, border:-border, border:-border].max()>0])
    except Exception as e:
        print('centers')
        print(e)
        center = np.array(mse_mean_mask_filter.shape) // 2 # center of volume
    return center

def get_center_by_mse_simple(model_inputs, threshold=0.95):
    """

    Parameters
    ----------
    model_inputs : 4d CMR as numpy ndarray with t,z,y,x

    Returns tuple (z,x,y)
    -------

    """
    import tensorflow as tf

    t, nz, ny, nx = model_inputs.shape
    include_slices = int(nz*.3)
    border_percentage = 20
    nyx = np.array([int(ny), int(nx)])
    border = ((nyx / 100) * border_percentage).astype(np.int)

    # take only the basal area and exclude areas close to the border
    model_inputs = model_inputs[:,:include_slices,border[0]:-border[0], border[1]:-border[1]]
    model_inputs = model_inputs[...,None]
    temp_roll = np.roll(model_inputs, shift=-1, axis=0)
    mse_ = tf.keras.metrics.mean_squared_error(model_inputs, temp_roll).numpy()

    mse_mean = np.mean(mse_, axis=0)
    thresh = np.quantile(mse_mean, threshold)
    # find the focus points f of deformation based on a percentile threshold
    f_z, f_x, f_y = np.where(mse_mean > thresh)
    f_z, f_x, f_y= f_z.mean(), f_y.mean(), f_x.mean()
    return np.array([f_y, f_x]) + border


def get_first_idx(mask, min_slices_labelled=3):
    '''
    Get the index of the first labelled timestep of a 4D mask
    Parameters
    ----------
    mask : np.ndarray with t,z,x,y,c

    Returns (int) idx of the first labelled t
    -------

    '''
    for i, mask3d in enumerate(mask):
        # check if we have more than n = 3 slices labelled
        masked_slices = np.where(mask3d.sum(axis=(1, 2)) > 0)[0]  # this is a tuple, we need the first element
        if len(masked_slices) > min_slices_labelled:
            break
    return i

def get_ip_from_2dmask(nda, debug=False, rev=False):
    """
    Find the RVIP on a 2D mask with the following labels
    RV (0), LVMYO (1) and LV (2) mask

    Parameters
    ----------
    nda : numpy ndarray with one hot encoded labels
    debug :

    Returns a tuple of two points anterior IP, inferior IP, each with (y,x)-coordinates
    -------

    """
    if debug: print('msk shape: {}'.format(nda.shape))
    # initialise some values
    first, second = None, None
    # find first and second insertion points
    myo_msk = (nda == 2).astype(np.uint8)
    comb_msk = ((nda == 1) | (nda == 2) | (nda == 3)).astype(np.uint8)
    myo_contours, hierarchy = cv2.findContours(myo_msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    comb_contours, hierarchy = cv2.findContours(comb_msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(myo_contours) > 0 and len(comb_contours) > 0: # we just need to search for IP if there are two contours
        # some lambda helpers
        # transform and describe contour lists to pythonic list which makes "elem in" syntax possible
        clean_contour = lambda cont: list(map(lambda x: (x[0][0], x[0][1]), cont[0]))
        descr_cont = lambda cont: print(
            'len: {}, first elem: {}, type of one elem: {}'.format(len(cont), cont[0], type(cont[0])))

        # clean/describe both contours
        myo_clean = clean_contour(myo_contours)
        if debug: descr_cont(myo_clean)
        comb_clean = clean_contour(comb_contours)
        if debug: descr_cont(comb_clean)

        # initialise some values
        septum_visited = False
        border_visited = False
        memory_first = None
        for p in myo_clean:
            if debug: print('p {} in {}'.format(p, p in comb_clean))
            # we are at the border,
            # moving anti-clockwise,
            # we dont know if we are in the septum
            # no second IP found so far.

            if p in comb_clean:
                border_visited = True
                if septum_visited and not second:
                    # take the first point after the septum as second IP
                    # we are at the border
                    # we have been at the septum
                    # no second defined so far
                    second = p
                    if debug: print('second= {}'.format(second))

                # we are at the border
                if not first:
                    # if we haven't been at the septum, update/remember this point
                    # use the last visited point before visiting the septum as first IP
                    memory_first = p
                    if debug: print('memory= {}'.format(memory_first))
            else:
                septum_visited = True  # no contour points matched --> we are at the septum
                if border_visited and not first:
                    first = memory_first
        if second and not first: # if our contour started at the first IP
            first = memory_first
        #assert first and second, 'missed one insertion point: first: {}, second: {}'.format(first, second)
        if debug: print('first IP: {}, second IP: {}'.format(first, second))
    if rev: first, second = (first[1], first[0]), (second[1], second[0])

    return first, second


def calc_resampled_size(sitk_img, target_spacing):
    if type(target_spacing) in [list, tuple]:
        target_spacing = np.array(target_spacing)
    old_size = np.array(sitk_img.GetSize())
    old_spacing = np.array(sitk_img.GetSpacing())
    logging.debug('old size: {}, old spacing: {}, target spacing: {}'.format(old_size, old_spacing,
                                                                             target_spacing))
    new_size = (old_size * old_spacing) / target_spacing
    return list(np.around(new_size).astype(np.int))


def match_2d_on_nd(nda, avg):
    if nda.ndim == 2:
        return match_2d_hist_on_2d(nda, avg)
    elif nda.ndim == 3:
        return match_2d_hist_on_3d(nda, avg)
    elif nda.ndim == 4:
        return match_2d_hist_on_4d(nda, avg)
    else:
        logging.info('shape for histogram matching does not fit to any method, return unmodified nda')
        return nda

from skimage import exposure
def match_2d_hist_on_2d(nda, avg):
    return skimage.exposure.match_histograms(nda, avg)


def match_2d_hist_on_3d(nda, avg):
    for z in range(nda.shape[0]):
        nda[z] = skimage.exposure.match_histograms(nda[z], avg, multichannel=False)
    return nda


def match_2d_hist_on_4d(nda, avg):
    for t in range(nda.shape[0]):
        for z in range(nda.shape[1]):
            nda[t, z] = skimage.exposure.match_histograms(nda[t, z], avg, multichannel=False)
    return nda
