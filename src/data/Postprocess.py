import numpy as np
from skimage import measure
import logging
import SimpleITK as sitk


import tensorflow as tf
import scipy.interpolate
from scipy.interpolate import interp1d
from scipy import ndimage
from src.models.KerasLayers import get_angle_tf,get_focus_tf,get_idxs_tf, flow2direction_lambda, minmax_lambda





def align_resample_multi(nda_vects, gt, gt_len, mask_by_norm=True,norm_percentile = 70,target_t = 30, normalise_dir=True, normalise_norm=True):
    """
    Alignment wrapper for a full dataset
    align norm and direction by the cardiac phase ED
    resample all 1D feature vectors to the same length and min/max normalise into [0,1] and [-1,1]
    this should help to validate the pre-defined rules, detect outliers, and if both together explains the cardiac phases
    Args:
        nda_vects ():
        gt ():
        gt_len ():
        target_t ():

    Returns:

    """

    dir_ndas, norm_ndas, indices = [],[],[]

    number_of_patients = nda_vects.shape[0]

    for p in range(number_of_patients):
        if p % 20 == 0: print('processing patient : {}'.format(p))
        cardiac_cycle_length = int(gt_len[p, :, 0].sum())
        ind = np.argmax(gt[p][:cardiac_cycle_length], axis=0)
        gt_onehot = gt[p][:cardiac_cycle_length]
        deformable_nda = nda_vects[p,:cardiac_cycle_length]
        dir_, norm_, ind_ = align_and_resample(cardiac_cycle_length=cardiac_cycle_length,
                                               ind=ind,
                                               gt_onehot=gt_onehot,
                                               deformable_nda=deformable_nda,
                                               mask_by_norm=mask_by_norm,
                                               norm_percentile = norm_percentile,
                                               target_t = target_t,
                                               normalise_dir=normalise_dir,
                                               normalise_norm=normalise_norm)
        dir_ndas.append(dir_)
        norm_ndas.append(norm_)
        indices.append(ind_)

    return np.stack(dir_ndas,axis=0), np.stack(norm_ndas,axis=0), np.stack(indices,axis=0)


def align_and_resample(cardiac_cycle_length, ind, gt_onehot, deformable_nda, mask_by_norm=True,norm_percentile = 70,target_t = 30, normalise_dir=True, normalise_norm=True):
    """
    Align norm and direction by the cardiac phase ED
    resample all 1D feature vectors to the same length and min/max normalise into [0,1] and [-1,1]
    this should help to validate the pre-defined rules, detect outliers, and if both together explains the cardiac phases
    Args:
        cardiac_cycle_length ():
        ind ():
        gt_onehot ():
        deformable_nda ():
        mask_by_norm ():
        norm_percentile ():
        target_t ():
        normalise_dir ():
        normalise_norm ():

    Returns:

    """
    # define some central params
    lower, mid, upper = -1, 0, 1
    xval = np.linspace(0, 1, target_t)

    dir_axis = 0
    gt_ed = ind[0]

    # magnitude/norm as mean
    norm_full = np.linalg.norm(deformable_nda, axis=-1)

    ########################## mask norm and direction by motion around the myocard/strongest motion over time
    norm_msk = norm_full.copy()
    norm_msk = np.mean(norm_msk, axis=0)
    norm_msk = norm_msk > np.percentile(norm_msk, norm_percentile)
    if mask_by_norm: norm_full = norm_full * norm_msk

    # balanced center, move the volume center towards the greatest motion
    dim_ = deformable_nda.shape[1:-1]
    ct = ndimage.center_of_mass(norm_msk)
    ct_center = np.array(dim_) // 2
    ct = (ct + ct_center) // 2
    idx = get_idxs_tf(dim_)
    c = get_focus_tf(ct, dim_)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    # direction
    directions = flow2direction_lambda([deformable_nda[...,dir_axis:], centers_tensor[...,dir_axis:]])[..., 0]
    if mask_by_norm: directions = directions * norm_msk
    # direction mean
    directions = np.average(directions, axis=(1, 2, 3), weights=np.abs(directions)>0.01)
    #directions = np.mean(directions, axis=(1, 2, 3))
    # direction ed aligned
    directions = np.roll(directions, -1 * gt_ed)
    # direction interpolate to unique length
    f = interp1d(np.linspace(0, 1, directions.shape[0]), directions, kind='linear')
    directions = f(xval)
    # direction min/max normalised between -1,1
    if normalise_dir: directions = minmax_lambda([directions,lower,upper])

    norm_nda = norm_full.mean(axis=(1, 2, 3))
    # norm ed aligned
    norm_nda = np.roll(norm_nda, -1 * gt_ed)
    # interpolate to unique length
    norm_nda = np.interp(xval, np.linspace(0, 1, norm_nda.shape[0]), norm_nda)
    # norm min/max aligned to 0,1
    if normalise_norm: norm_nda = minmax_lambda([norm_nda,mid,upper])

    # roll, scale, round and clip the gt indicies, to get an aligned distribution of the labels
    gt_onehot_rolled = np.roll(gt_onehot, -1 * gt_ed, axis=0)
    resize_factor = target_t / cardiac_cycle_length
    gt_onehot_rolled = np.argmax(gt_onehot_rolled, axis=0)
    gt_onehot_rolled = np.clip(np.rint(gt_onehot_rolled * resize_factor), a_min=0, a_max=target_t - 1)

    return directions, norm_nda, gt_onehot_rolled


def undo_generator_steps(ndarray, cfg, interpol=sitk.sitkLinear, orig_sitk=None):
    """
    Undo the generator steps for a 3D volume
    1. calculate the intermediate size (which was used in the generator)
    2. Pad and crop to the intermediate size
    3. set the spacing given by the config
    4. Resample to the original spacing
    Parameters
    ----------
    ndarray :
    p :
    cfg :
    interpol :
    orig_sitk :

    Returns
    -------

    """
    from src.data.Preprocess import resample_3D, pad_and_crop
    from src.data.Preprocess import calc_resampled_size

    orig_size_ = orig_sitk.GetSize()
    orig_spacing_ = orig_sitk.GetSpacing()
    logging.debug('original shape: {}'.format(orig_size_))
    logging.debug('original spacing: {}'.format(orig_spacing_))

    # numpy has the following order: h,w,c (or z,h,w,c for 3D)
    w_h_size_sitk = orig_size_
    w_h_spacing_sitk = orig_spacing_

    # calculate the size of the image before crop or pad
    # z, x, y -- after reverse --> y,x,z we set this spacing to the input nda before resampling
    cfg_spacing = np.array((orig_spacing_[-1], *cfg['SPACING']))
    cfg_spacing = list(reversed(cfg_spacing))
    new_size = calc_resampled_size(orig_sitk, cfg_spacing)
    new_size = list(reversed(new_size))

    # pad, crop to original physical size in current spacing
    logging.debug('pred shape: {}'.format(ndarray.shape))
    logging.debug('intermediate size after pad/crop: {}'.format(new_size))

    ndarray = pad_and_crop(ndarray, new_size)
    logging.debug(ndarray.shape)

    # resample, set current spacing
    img_ = sitk.GetImageFromArray(ndarray)
    img_.SetSpacing(tuple(cfg_spacing))
    img_ = resample_3D(img_, size=w_h_size_sitk, spacing=w_h_spacing_sitk, interpolate=interpol)

    logging.debug('Size after resampling into original spacing: {}'.format(img_.GetSize()))
    logging.debug('Spacing after undo function: {}'.format(img_.GetSpacing()))

    return img_

def clean_3d_prediction_3d_cc(pred):
    """
    Find the biggest connected component per label
    This is a debugging method, which will plot each step
    returns: a tensor with the same shape as pred, but with only one cc per label
    """

    # avoid labeling images with float values
    assert len(np.unique(pred)) < 10, 'to many labels: {}'.format(len(np.unique(pred)))

    cleaned = np.zeros_like(pred)

    def clean_3d_label(val):

        """
        has access to pred, no passing required
        """

        # create a placeholder
        biggest = np.zeros_like(pred)
        biggest_size = 0

        # find all cc for this label
        # tensorflow operation is only in 2D
        # all_labels = tfa.image.connected_components(np.uint8(pred==val)).numpy()
        all_labels = measure.label(np.uint8(pred == val), background=0)

        for c in np.unique(all_labels)[1:]:
            mask = all_labels == c
            mask_size = mask.sum()
            if mask_size > biggest_size:
                biggest = mask
                biggest_size = mask_size
        return biggest

    for val in np.unique(pred)[1:]:
        biggest = clean_3d_label(val)
        cleaned[biggest] = val
    return cleaned
import cv2

def clean_3d_prediction_2d_cc(pred):
    cleaned = []
    # for each slice
    for s in pred:
        new_img = np.zeros_like(s)  # step 1
        # for each label
        for val in np.unique(s)[1:]:  # step 2
            mask = np.uint8(s == val)  # step 3
            labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]  # step 4
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # step 5
            new_img[labels == largest_label] = val  # step 6
        cleaned.append(new_img)
    return np.stack(cleaned, axis=0)