# predict cardiac phases for a cv experiment
import logging
import os


def predict(cfg_file, data_root='', c2l=False):
    """
    Predict on the held-out validation split
    Parameters
    ----------
    cfg_file :
    data_root :
    c2l :

    Returns
    -------

    """
    import json, logging, os
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    import tensorflow as tf
    tf.get_logger().setLevel('FATAL')
    from logging import info
    import numpy as np
    from src.data.Dataset import get_trainings_files
    from src.utils.Utils_io import Console_and_file_logger, ensure_dir
    from src.data.PhaseGenerators import PhaseRegressionGenerator_v2
    from src.models.PhaseRegModels import PhaseRegressionModel
    from ProjectRoot import change_wd_to_project_root
    change_wd_to_project_root()


    from src.utils.Tensorflow_helper import choose_gpu_by_id


    # load the experiment config
    if type(cfg_file) == type(''):
        with open(cfg_file, encoding='utf-8') as data_file:
            config = json.loads(data_file.read())
    else:
        config = cfg_file
    globals().update(config)

    # ------------------------------------------define GPU id/s to use
    GPU_IDS = config.get('GPU_IDS', '0,1')
    GPUS = choose_gpu_by_id(GPU_IDS)
    print(GPUS)
    print(tf.config.list_physical_devices('GPU'))

    EXPERIMENT = config.get('EXPERIMENT', 'UNDEFINED')
    Console_and_file_logger(EXPERIMENT, logging.INFO)
    info('Loaded config for experiment: {}'.format(EXPERIMENT))

    # Load SAX volumes
    # cluster to local data mapping
    if c2l:
        config['DATA_PATH_SAX'] = os.path.join(data_root, 'sax')
        config['DF_FOLDS'] = os.path.join(data_root, 'df_kfold.csv')
        config['DF_META'] = os.path.join(data_root, 'SAx_3D_dicomTags_phase.csv')

    x_train_sax, y_train_sax, x_val_sax, y_val_sax = get_trainings_files(data_path=config['DATA_PATH_SAX'],
                                                                         path_to_folds_df=config['DF_FOLDS'],
                                                                         fold=config['FOLD'])
    logging.info('SAX train CMR: {}, SAX train masks: {}'.format(len(x_train_sax), len(y_train_sax)))
    logging.info('SAX val CMR: {}, SAX val masks: {}'.format(len(x_val_sax), len(y_val_sax)))

    # turn off all augmentation operations while inference
    # create another config for the validation data
    # we want the prediction to run with batchsize of 1
    # otherwise we might inference only on the even number of val files
    # the mirrored strategy needs to get a single gpu instance named, otherwise batchsize=1 does not work
    val_config = config.copy()
    val_config['SHUFFLE'] = False
    val_config['AUGMENT'] = False
    val_config['AUGMENT_PHASES'] = False
    val_config['AUGMENT_TEMP'] = False
    val_config['BATCHSIZE'] = 1
    val_config['HIST_MATCHING'] = False
    val_config['GPUS'] = ['/gpu:0']
    validation_generator = PhaseRegressionGenerator_v2(x_val_sax, x_val_sax, config=val_config)

    model = PhaseRegressionModel(val_config).get_model()
    logging.info('Trying to load the model weights')
    logging.info('work dir: {}'.format(os.getcwd()))
    logging.info('model weights dir: {}'.format(os.path.join(val_config['MODEL_PATH'], 'model.h5')))
    model.load_weights(os.path.join(val_config['MODEL_PATH'], 'model.h5'))
    logging.info('loaded model weights as h5 file')

    # predict on the validation generator
    preds, moved, vects = model.predict(validation_generator)
    logging.info('Shape of the predictions: {}'.format(preds.shape))

    # get all ground truth vectors, each y is a list with [onehot,moved, zeros]
    gts = np.concatenate([y[0] for x, y in validation_generator],axis=0)
    logging.info('Shape of GT: {}'.format(gts.shape))

    pred_path = os.path.join(val_config['EXP_PATH'], 'pred')
    moved_path = os.path.join(val_config['EXP_PATH'], 'moved')
    ensure_dir(pred_path)
    ensure_dir(moved_path)
    pred_filename = os.path.join(pred_path, 'gtpred_fold{}.npy'.format(val_config['FOLD']))
    moved_filename = os.path.join(moved_path, 'moved_f{}.npy'.format(val_config['FOLD']))
    vects_filename = os.path.join(moved_path, 'vects_f{}.npy'.format(val_config['FOLD']))
    np.save(pred_filename, np.stack([gts, preds], axis=0))
    np.save(moved_filename, moved)
    np.save(vects_filename, vects)

    patients_filename = os.path.join(pred_path, 'patients.txt')
    with open(patients_filename, "a+") as f:
        _ = [f.write(str(val_config['FOLD']) + '_' + os.path.basename(elem) +'\n') for elem in x_val_sax]
    logging.info('saved as: \n{}\n{} \ndone!'.format(pred_filename, patients_filename))


def detect_phases(dir_1d_mean):
    """
    Detect five cardiac phases from a 1D direction curve
    Args:
        dir_1d_mean (): np.ndarray() with shape t,

    Returns:

    """
    import scipy.signal as sig
    import numpy as np

    length = len(dir_1d_mean)

    # MS
    # Global min of f(x)
    ms = np.argmin(dir_1d_mean)
    ms = ms - 1  # take the bucket before the first min peak

    # ES
    # First time f(x)>0 after MS
    ms_idx = ms + 1
    cycle = np.concatenate([dir_1d_mean[ms_idx:], dir_1d_mean[:ms_idx]])
    cycle = cycle[:np.argmax(cycle)]  # ES should be between MS and approx PF == argmax
    temp_ = 0
    es_found = False
    negative_slope = False
    for idx, elem in enumerate(cycle):
        if elem < 0:
            negative_slope = True
            # temp_ = idx
        elif elem >= 0 and negative_slope:
            es_found = True
            temp_ = idx
            negative_slope = False
            # break # stop after first zero-transition
    if es_found:
        es = ms_idx + temp_
        es = es - 1
    else:
        es = ms_idx  # the frame after ms, fallback
    if es >= length:
        es = np.mod(es, length)
        print('ES overflow: {}, ms:{}'.format(es, ms))

    # PF
    # First peak after ES
    es_idx = es + 1
    seq = np.concatenate([dir_1d_mean[es_idx:], dir_1d_mean[:es_idx]])
    peaks, prop = sig.find_peaks(seq)  # height=0.6 we normalise between -1 and 1, PF should be close to argmax
    if len(peaks > 0):
        pf_idx = es_idx + peaks[0]  # take the peak after es
        pf = pf_idx - 1
    else:
        print('pf not clear, set to ES {} + 1'.format(es))
        pf = es_idx

    # sometimes the relaxation after the MD phase is stronger than the PF relaxation
    # otherwise we could simply:
    # pf = np.argmax(dir_1d_mean)

    pf = np.mod(pf, length)

    # ED
    # Between pf and ms: last time f(x) cross zero from positive to negative
    # a priori knowledge ED needs a minimal distance of 2 frames towards MS
    # CHANGED the minimal distance between ED and MS
    cycle = np.concatenate([dir_1d_mean[pf_idx:], dir_1d_mean[:ms]])
    # print(cycle)
    ed_found = False
    last_idx_positive = True  # we start at the pf, which is the peak(dir)
    for idx, elem in enumerate(cycle):

        # this enables to find the last transition from pos to neg
        if elem >= 0:
            last_idx_positive = True
        # remember the last idx before the direction gets negative the last time before ms
        elif elem < 0 and last_idx_positive:  # first time direction negative
            ed_found = True  # for fallbacks
            temp_ = idx  # idx before negative direction
            # print('found transition at: {}'.format(idx))
            last_idx_positive = False  # remember only the first idx after transition

    if ed_found:
        ed = pf_idx + temp_
        # print('ed:{}, pf:{}, temp_:{}, lenght: {}'.format(ed,pf,temp_,length))
    else:
        # if we dont find a transition from positive to negative, take the idx which is the closest to zero
        temp_ = np.argmin(np.abs(cycle))  # make sure we have a minimal distance
        ed = pf_idx + temp_
        print('ED: no transition found between {}-{} , closest id to 0: {}, ed = {}'.format(pf, ms, temp_, ed))

    ed = np.mod(ed, length)
        # MD
    # Middle between PF and ED
    ed_slice_idx = ed
    if ed_slice_idx <= pf_idx:  # ed overflow --> beginning of cmr stack
        ed_slice_idx = length + ed
    md = (pf_idx + ed_slice_idx) // 2  # the bucket after the middle
    md = md + 2
    md = np.mod(md, length)

    '''seq = np.concatenate([dir_1d_mean[pf_idx:], dir_1d_mean[:ed]])
    peaks, prop = sig.find_peaks(seq)  # height=0.6 we normalise between -1 and 1, PF should be close to argmax
    if len(peaks > 0):
        md = pf_idx + peaks[0]
        #print('peak: {}'.format(peaks[-1]))
    elif ed > pf_idx:
        md = pf_idx + (ed-pf_idx)//2
    else:
        md = pf_idx + (length-pf_idx + ed)//2
    md = np.mod(md, length)'''

    return np.array([ed, ms, es, pf, md])


def get_phases_from_vects(vects_nda, length=-1, plot=True, dir_axis=0, gtind=[], norm_percentile=0., exp_path=None, patient='temp',
                          save=False):
    import scipy.signal
    import scipy.signal as sig
    import scipy.ndimage
    from scipy.ndimage import gaussian_filter1d
    global centers_tensor
    from scipy import ndimage
    import numpy as np
    from src.data.Preprocess import clip_quantile
    from src.models.KerasLayers import minmax_lambda, get_idxs_tf, get_focus_tf, flow2direction_lambda
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from src.visualization.Visualize import show_2D_or_3D


    lower, mid, upper = -1,0,1

    dim_z = vects_nda.shape[1]
    z = dim_z // 2

    # norm of the vector
    norm_nda = np.linalg.norm(vects_nda[..., dir_axis:], axis=-1)
    norm_nda = clip_quantile(norm_nda, 0.99)
    norm_nda = minmax_lambda([norm_nda, mid, upper])
    norm_msk = norm_nda.copy()
    norm_msk = np.mean(norm_msk[:-1], axis=0)
    norm_msk = norm_msk > np.percentile(norm_msk, norm_percentile)
    # for norm msk improvements the following did not work well:
    # connected component filtering before COM
    # Gauss smoothing or any other conv operation such as closing etc.
    # usually there are occlusions that stop these methods to work for each patient

    norm_nda = norm_nda * norm_msk

    # balanced center, move the volume center towards the greatest motion (norm_msk)
    dim_ = vects_nda.shape[1:-1]
    ct = ndimage.center_of_mass(norm_msk) #x,y =  np.mean(np.where(norm_mask)) compareable results, and usable in a differentiable model with tf
    ct_center = np.array(dim_) // 2
    ct = (ct + ct_center) // 2
    idx = get_idxs_tf(dim_)
    c = get_focus_tf(ct, dim_)
    centers = c - idx
    centers_tensor = centers[tf.newaxis, ...]

    ######################################## new norm mask according to the mse filtering - 29.03

    #nda_1d_mean = np.nanmean(norm_nda, axis=(1, 2, 3))
    nda_1d_mean = np.average(norm_nda, axis=(1, 2, 3), weights=np.broadcast_to(norm_msk, shape=norm_nda.shape))
    nda_1d_max = np.max(norm_nda, axis=(1, 2, 3))
    nda_1d_median = np.median(norm_nda, axis=(1, 2, 3))

    # direction relative to the focus point C_x
    directions = flow2direction_lambda([vects_nda, centers_tensor])[..., 0].numpy()
    directions = directions * norm_msk
    directions = minmax_lambda([directions, lower, upper])

    dir_1d_mean = np.average(directions, axis=(1, 2, 3), weights=np.broadcast_to(norm_msk, shape=directions.shape)) # np.abs(directions)>0.01
    #dir_1d_mean = np.nanmean(directions, axis=(1, 2, 3))
    dir_1d_median = np.median(directions, axis=(1, 2, 3))

    # Find phases
    # 1st smooth the direction with a rolling averge (kernelsize=4)
    # or via gaussian filter
    # 2nd min/max normalise each direction vector in a range [-1:1]
    dir_1d_mean = gaussian_filter1d(dir_1d_mean, sigma=2, mode='wrap')
    dir_1d_mean = minmax_lambda([dir_1d_mean, lower, upper])
    ind = detect_phases(dir_1d_mean=dir_1d_mean[:length])# we dont need to provide the length as parameter

    # plot the mean/max norm for one patient over time
    if plot:
        import seaborn as sb
        #with sb.plotting_context("paper"):
        plt.rcParams['font.size'] = '16'
        figsize = (25, 2)
        fig = plt.figure(figsize=figsize)

        # DIR 2D+t
        dir_2d_t = directions[:, z]
        div_cmap = 'bwr'
        fig = show_2D_or_3D(dir_2d_t, allow_slicing=False, f_size=(25, 2), cmap=div_cmap, fig=fig)
        ax_ = fig.get_axes()[0]
        _ = ax_.set_yticks([])
        _ = ax_.set_xticks([])
        ax = fig.get_axes()[1]
        _ = ax.set_ylabel(r'$\alpha$ ' + '\n2d+t\nmid')
        _ = ax.set_yticks([])
        _ = ax.set_xticks([])
        #cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        #fig.colorbar(ax.get_images()[len(ax.get_images()) // 2], cax=cax, orientation='horizontal')
        #cbaxes = fig.add_axes([0.1, 0.1, 0.03, 0.8])
        fig.colorbar(ax.get_images()[-1],ax=[fig.get_axes()[-1]],fraction=0.046, pad=0.04,shrink=0.8).set_ticks([-1,0,1])

        #
        figsize = (25, 2)
        rows = 2
        pos = 2
        ax = fig.add_subplot(rows, 1, pos)

        # DIR 2D T x Z
        directions_tz = minmax_lambda([directions.mean(axis=(2, 3)), lower, upper])
        _ = ax.imshow(directions_tz.T, aspect='auto', cmap=div_cmap, vmin=-1,vmax=1,origin='lower', interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        ax2 = ax.twinx()
        _ = ax2.plot(dir_1d_mean, c='black', label='dir 1d+t')
        _ = ax2.set_ylabel(r'$\alpha_{t}$')
        _ = ax.set_yticks([])
        _ = ax.set_ylabel(r'$\alpha$' + '\nz+t\nap:ba')
        ax2.label_outer()
        if save: fig.savefig(os.path.join(exp_path, '{}_alpha.svg'.format(patient)))

        norm_cmap = 'hot'

        # NORM 2D + t
        fig = plt.figure(figsize=figsize)
        norm_2d_t = norm_nda[:, z]
        norm_2d_t = minmax_lambda([norm_2d_t, mid, upper])
        fig = show_2D_or_3D(norm_2d_t, allow_slicing=False, f_size=(25, 2), cmap=norm_cmap, interpolation='none',
                            fig=fig)
        ax = fig.get_axes()[0]
        _ = ax.set_yticks([])
        _ = ax.set_xticks([])
        ax = fig.get_axes()[1]
        _ = ax.set_ylabel(r'$|\vec{v}|$' + '\n2d+t\nmid')
        _ = ax.set_yticks([])
        _ = ax.set_xticks([])
        #cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        #fig.colorbar(ax.get_images()[0], cax=cax, orientation='horizontal')
        fig.colorbar(ax.get_images()[-1], ax=[fig.get_axes()[-1]], fraction=0.046, pad=0.04, shrink=0.6).set_ticks([0,1])

        # print(len(fig.get_axes()))
        figsize = (25, 2)
        rows = 2
        pos = 2
        ax = fig.add_subplot(rows, 1, pos)

        # NORM 2D TxZ
        norm2d = minmax_lambda([norm_nda.mean(axis=(2, 3)), mid, upper])
        _ = ax.imshow(norm2d.T, aspect='auto', origin='lower', cmap=norm_cmap, interpolation='none')
        _ = ax.set_xticks(gtind, minor=False)
        _ = ax.set_yticks([])
        _ = ax.set_ylabel(r'$|\vec{v}|$' + '\nz+t\nap:ba')
        ax2 = ax.twinx()
        _ = ax2.plot(minmax_lambda([nda_1d_mean, mid, upper]), c='black', label='norm 1d+t')
        _ = ax2.set_ylabel(r'$|\vec{v}_{t}|$')
        # print(len(fig.get_axes()))
        if save: fig.savefig(os.path.join(exp_path, '{}_norm.svg'.format(patient)))
        return fig, ind
    else:
        return ind

def predict_phase_from_deformable(exp_path, create_figures=True,norm_thresh=70, dir_axis=1):
    """
    Predict the temporal occurence for five cardiac phases from a cmr-phase-regression experiment folder
    Expects to find all files written from a CV-experiment, e.g.> train_regression_model.py
    Args:
        exp_path (str): full path to a phase regression experiment
        norm_thresh (int): 0 < norm_thresh < 100
        dir_axis (int): out of [0,1], 0 = z,y,x motion, 1 = y,x motion, z- is negative during systole, y,x positive

    Returns:

    """
    import numpy as np
    import pandas as pd
    import os
    import logging
    from src.data.Dataset import load_phase_reg_exp
    from src.utils.Metrics import meandiff

    # load all files
    # load all files of this experiment
    nda_vects, gt, pred, gt_len, mov, patients = load_phase_reg_exp(exp_path)

    logging.info('files loaded, continue with deformable2direction2phase')

    # predict phase per patient and write result as df into experiment folder
    pred_u = np.zeros_like(gt)
    upred_ind = []
    gt_ind = []
    cycle_len = []
    print(pred_u.shape)
    for i in range(pred_u.shape[0]):
        weight = 1
        cardiac_cycle_length = int(gt_len[i, :, 0].sum())
        cycle_len.append(cardiac_cycle_length)
        ind = np.argmax(gt[i][:cardiac_cycle_length], axis=0)
        indices = get_phases_from_vects(vects_nda=nda_vects[i][:cardiac_cycle_length],
                                        length=cardiac_cycle_length,
                                        gtind=ind,
                                        plot=False,
                                        dir_axis=dir_axis,
                                        norm_percentile=norm_thresh)
        upred_ind.append(indices)
        gt_ind.append(ind)
        indices = np.array(indices)
        onehot = np.zeros((indices.size, cardiac_cycle_length))
        onehot[np.arange(indices.size), indices] = weight
        pred_u[i][0:cardiac_cycle_length] = onehot.T
    upred_ind = np.stack(upred_ind, axis=0)
    gt_ind = np.stack(gt_ind, axis=0)
    cycle_len = np.stack(cycle_len, axis=0)
    # re-create a compatible shape for the metric fn
    gt_ = np.stack([gt, gt_len], axis=1)
    pred_ = np.stack([pred_u, np.zeros_like(pred_u)], axis=1)

    # create some dataframes for further processing

    phases = ['ED', 'MS', 'ES', 'PF', 'MD']
    res = meandiff(gt_, pred_, apply_sum=False, apply_average=False)
    df_pfd = pd.DataFrame(res.numpy(), columns=phases)
    df_pfd['patient'] = patients
    df_pfd.to_csv(os.path.join(exp_path, 'cfd.csv'))
    # save predicted phases as csv
    pred_df = pd.DataFrame(upred_ind, columns=phases)
    pred_df['patient'] = patients
    pred_df.to_csv(os.path.join(exp_path, 'pred_phases.csv'))

    gt_df = pd.DataFrame(gt_ind, columns=phases)
    gt_df['patient'] = patients
    gt_df.to_csv(os.path.join(exp_path, 'gt_phases.csv'))

    # create some plots
    if create_figures:
        from src.data.Postprocess import align_resample_multi
        dirs, norms, gt_ind_scaled = align_resample_multi(nda_vects=nda_vects, gt=gt, gt_len=gt_len)

        from src.visualization.Visualize import plot_dir_norm,  plot_dir_norm_split_by, plot_pfd_per_phase_as_violin
        _ = plot_dir_norm(dirs, norms, gt_ind_scaled, exp_path)

        from src.data.Dataset import merge_patient_with_metadata
        for dataset in ['acdc', 'tof']:
            try:
                df_merge = merge_patient_with_metadata(patients=patients, dataset=dataset)
            except Exception as e:
                logging.info('found no metadata for the dataset: {}'.format(dataset))
        _ = plot_dir_norm_split_by(dirs, norms, gt_ind_scaled, df_merge=df_merge, split_by='target',
                               exp_path=exp_path)

        _ = plot_pfd_per_phase_as_violin(pred_df=pred_df, gt_df=gt_df, df_pfd=df_pfd, exp_path=exp_path)



    return pred_df, gt_df, df_pfd, res, cycle_len



if __name__ == "__main__":
    import argparse, os, sys, glob

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    parser = argparse.ArgumentParser(description='predict a phase registration model')

    # usually the exp root parameters should yield to a config, which encapsulate all experiment parameters
    parser.add_argument('-exp_root', action='store', default='/mnt/sds/sd20i001/sven/code/exp/miccai_baseline')
    parser.add_argument('-data', action='store', default='/mnt/ssd/data/gcn/02_imported_4D_unfiltered')
    parser.add_argument('-work_dir', action='store', default='/mnt/ssd/git/dynamic-cmr-models')
    parser.add_argument('-c2l', action='store_true', default=False)

    results = parser.parse_args()
    os.chdir(results.work_dir)
    sys.path.append(os.getcwd())
    print('given parameters: {}'.format(results))

    # get all cfgs - we expect to find 4 as we usually train a 4-fold cv
    # call the predict_fn for each cfg
    initial_search_pattern = 'config/config.json' # path to one experiment
    search_path = os.path.join(results.exp_root, initial_search_pattern)
    cfg_files = sorted(glob.glob(search_path))
    if len(cfg_files) == 0:
        # we called this script with the experiment root,
        # search for the sub-folders per split
        search_pattern = '**/config/config.json'
        search_path = os.path.join(results.exp_root, search_pattern)
        print(search_path)
        cfg_files = sorted(glob.glob(search_path))
        assert len(cfg_files) == 4, 'Expect 4 cfgs, but found {}'.format(len(cfg_files)) # avoid loading too many cfgs
    print(cfg_files)

    for cfg in cfg_files:
        try:
            predict(cfg_file=cfg, data_root=results.data, c2l=results.c2l)
        except Exception as e:
            print(e)
    exit()
