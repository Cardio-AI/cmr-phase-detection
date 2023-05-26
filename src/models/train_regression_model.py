


def train_fold(config, in_memory=False):
    # make sure necessary config params are given, otherwise replace with default
    import tensorflow
    import tensorflow as tf
    import numpy as np
    tf.get_logger().setLevel('FATAL')
    tf.random.set_seed(config.get('SEED', 42))
    np.random.seed(config.get('SEED', 42))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    from src.utils.Tensorflow_helper import choose_gpu_by_id
    # ------------------------------------------define GPU id/s to use
    GPU_IDS = config.get('GPU_IDS', '0,1')
    GPUS = choose_gpu_by_id(GPU_IDS)
    print(GPUS)
    print(tf.config.list_physical_devices('GPU'))
    # ------------------------------------------ import helpers
    #from tensorflow.python.client import device_lib
    # import external libs
    from time import time
    import logging, os

    # local imports
    from src.utils.Utils_io import Console_and_file_logger, init_config, ensure_dir
    from src.utils.KerasCallbacks import get_callbacks
    from src.data.Dataset import get_trainings_files, all_files_in_df
    from src.data.PhaseGenerators import PhaseRegressionGenerator_v2
    from src.models.PhaseRegModels import PhaseRegressionModel

    # make all config params known to the local namespace
    locals().update(config)

    # overwrite the experiment names and paths, so that each cv gets an own sub-folder
    EXPERIMENT = config.get('EXPERIMENT')
    FOLD = config.get('FOLD')

    EXPERIMENT = '{}f{}'.format(EXPERIMENT, FOLD)
    """timestemp = str(datetime.datetime.now().strftime(
        "%Y-%m-%d_%H_%M"))"""  # add a timestep to each project to make repeated experiments unique

    EXPERIMENTS_ROOT = 'exp/'
    EXP_PATH = config.get('EXP_PATH')
    FOLD_PATH = os.path.join(EXP_PATH, 'f{}'.format(FOLD))
    MODEL_PATH = os.path.join(FOLD_PATH, 'model', )
    TENSORBOARD_PATH = os.path.join(FOLD_PATH, 'tensorboard_logs')
    CONFIG_PATH = os.path.join(FOLD_PATH, 'config')

    ensure_dir(MODEL_PATH)
    ensure_dir(TENSORBOARD_PATH)
    ensure_dir(CONFIG_PATH)

    DATA_PATH_SAX = config.get('DATA_PATH_SAX')
    DF_FOLDS = config.get('DF_FOLDS', None)
    DF_META = config.get('DF_META', None)
    EPOCHS = config.get('EPOCHS', 100)

    Console_and_file_logger(path=FOLD_PATH, log_lvl=logging.INFO)
    config = init_config(config=locals(), save=True)
    logging.info('Is built with tensorflow: {}'.format(tf.test.is_built_with_cuda()))
    logging.info('Visible devices:\n{}'.format(tf.config.list_physical_devices()))
    #logging.info('Local devices: \n {}'.format(device_lib.list_local_devices()))

    # get k-fold data from DATA_ROOT and subdirectories
    x_train_sax, y_train_sax, x_val_sax, y_val_sax = get_trainings_files(data_path=DATA_PATH_SAX,
                                                                         path_to_folds_df=DF_FOLDS,
                                                                        fold=FOLD)

    """examples = 12
    x_train_sax, y_train_sax, x_val_sax, y_val_sax = x_train_sax[:examples], y_train_sax[:examples], x_val_sax[:examples], y_val_sax[:examples]"""
    #x_train_sax = [x for x in x_train_sax if '4a4pvcyl_2006' in x] * 4
    #x_val_sax = [x for x in x_val_sax if '38' in x] * 4
    #x_train_sax = x_val_sax
    logging.info('SAX train CMR: {}, SAX train masks: {}'.format(len(x_train_sax), len(y_train_sax)))
    logging.info('SAX val CMR: {}, SAX val masks: {}'.format(len(x_val_sax), len(y_val_sax)))

    t0 = time()
    # check if we find each patient in the corresponding dataframe
    if DF_META is not None and os.path.exists(DF_META):
       all_given = all_files_in_df(DF_META, x_train_sax, x_val_sax)
       logging.info('found all patients in df meta: {}'.format(all_given))

    debug = 0  # make sure single threaded
    # Create the batchgenerators
    config['BATCHSIZE'] = 2
    config['T_SHAPE'] = 50
    if debug:
        config['SHUFFLE'] = False
        config['WORKERS'] = 1
        config['BATCHSIZE'] = 1
    batch_generator = PhaseRegressionGenerator_v2(x_train_sax, x_train_sax, config=config, in_memory=in_memory)
    val_config = config.copy()
    val_config['AUGMENT'] = False
    val_config['AUGMENT_PHASES'] = False
    val_config['HIST_MATCHING'] = False
    val_config['AUGMENT_TEMP'] = False
    validation_generator = PhaseRegressionGenerator_v2(x_val_sax, x_val_sax, config=val_config, in_memory=in_memory)

    import matplotlib.pyplot as plt
    from src.visualization.Visualize import show_2D_or_3D

    if debug:
        path_ = 'data/interim/{}_focus_mse/'.format('tof_volume')
        ensure_dir(path_)
        i = 0
        for b in batch_generator:
            x,y = b
            x = x[0]
            for p in x:
                patient = os.path.basename(batch_generator.IMAGES[i]).split('_')[0]
                fig = show_2D_or_3D(p[0,...,0:1])
                plt.savefig('{}{}_{}.png'.format(path_,i,patient))
                plt.close()
                i = i+1
        i = 0
        for b in validation_generator:
            x,y = b
            x = x[0]
            for p in x:
                patient = os.path.basename(validation_generator.IMAGES[i]).split('_')[0]
                fig = show_2D_or_3D(p[0,...,0:1])
                plt.savefig('{}v{}_{}.png'.format(path_,i,patient))
                plt.close()
                i = i+1

    # get model
    #model = create_PhaseRegressionModel_v2(config)
    model = PhaseRegressionModel(config=config).get_model()

    # write the model summary to a txt file
    with open(os.path.join(FOLD_PATH, 'model_summary.txt'), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(line_length=140, print_fn=lambda x: fh.write(x + '\n'))

    # plot the model structure as graph
    tf.keras.utils.plot_model(
        model,
        show_dtype=False,
        show_shapes=True,
        to_file=os.path.join(FOLD_PATH, 'model.png'),
        show_layer_names=False,
        rankdir='TB',
        expand_nested=False,
        dpi=96
    )

    # training
    initial_epoch = 0
    #EPOCHS=1
    model.fit(
        x=batch_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=get_callbacks(config, batch_generator, validation_generator),
        initial_epoch=initial_epoch,
        #max_queue_size=config.get('QUEUE_SIZE',2),
        # use_multiprocessing=False,
        # workers=12,
        verbose=1)

    # free as much memory as possible
    import gc
    tf.keras.backend.clear_session()
    logging.info('Fold {} finished after {:0.3f} sec'.format(FOLD, time() - t0))
    del batch_generator
    del validation_generator
    del model
    gc.collect()
    return config


def main(args=None, in_memory=False):
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    # ------------------------------------------define logging and working directory
    # import the packages inside this function enables to train on different folds
    from ProjectRoot import change_wd_to_project_root
    change_wd_to_project_root()
    import sys, os, datetime, gc
    sys.path.append(os.getcwd())

    # local imports
    # import external libs
    #import tensorflow as tf
    #tf.get_logger().setLevel('FATAL')
    import cv2
    from src.utils.Utils_io import Console_and_file_logger, init_config

    EXPERIMENTS_ROOT = 'exp/'

    if args.cfg:
        import json
        cfg = args.cfg
        print('config given: {}'.format(cfg))
        # load the experiment config
        with open(cfg, encoding='utf-8') as data_file:
            config = json.loads(data_file.read())

        # Define new paths, so that we make sure that:
        # 1. we dont overwrite a previous config
        # 2. cluster based trainings are compatible with saving locally (cluster/local)
        # we dont need to initialise this config, as it should already have the correct format,
        # The fold configs will be saved within each fold run
        # add a timestep to each project to make repeated experiments unique
        EXPERIMENT = config.get('EXPERIMENT', 'UNDEFINED')
        timestemp = str(datetime.datetime.now().strftime("%Y-%m-%d_%H_%M"))

        config['EXP_PATH'] = os.path.join(EXPERIMENTS_ROOT, EXPERIMENT, timestemp)

        if args.data:  # if we specified a different data path (training from workspace or node temporal disk)
            config['DATA_PATH_SAX'] = os.path.join(args.data, "sax/")
            config['DF_FOLDS'] = os.path.join(args.data, "df_kfold.csv")
            config['DF_META'] = os.path.join(args.data, "SAx_3D_dicomTags_phase.csv")

        print(config)
    else:
        print('no config given, please select one from the  templates in exp/examples')

    from src.models.predict_phase_reg_model import predict
    for f in config.get('FOLDS', [0]):
        print('starting fold: {}'.format(f))
        config_ = config.copy()
        config_['FOLD'] = f
        cfg = train_fold(config_, in_memory=in_memory)
        predict(cfg)
        gc.collect()
        print('train fold: {} finished'.format(f))
    from src.models.evaluate_phase_reg import evaluate_supervised
    from src.models.predict_phase_reg_model import predict_phase_from_deformable
    evaluate_supervised(config.get('EXP_PATH'))
    predict_phase_from_deformable(config.get('EXP_PATH'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='train a phase registration model')

    # usually these two parameters should encapsulate all experiment parameters
    parser.add_argument('-cfg', action='store', default=None)
    parser.add_argument('-data', action='store', default=None)
    parser.add_argument('-inmemory', action='store', default=False) # enable in memory pre-processing on the cluster

    # anyway, there are cases were we want to define some specific parameters, a better choice would be to modify the config
    parser.add_argument('-sax', action='store', default='/mnt/ssd/data/gcn/02_imported_4D_unfiltered/sax/')
    parser.add_argument('-folds', action='store', default='/mnt/ssd/data/gcn/02_imported_4D_unfiltered/df_kfold.csv')
    parser.add_argument('-meta', action='store', default='/mnt/ssd/data/gcn/02_imported_4D_unfiltered/SAx_3D_dicomTags_phase.csv')
    parser.add_argument('-exp', action='store', default='temp_exp')
    parser.add_argument('-add_lstm', action='store_true', default=False)
    parser.add_argument('-lstm_units', action='store', default=64, type=int)
    parser.add_argument('-depth', action='store', default=4, type=int)
    parser.add_argument('-filters', action='store', default=20, type=int)
    parser.add_argument('-aug', action='store', default=True)
    parser.add_argument('-paug', action='store', default=False)
    parser.add_argument('-prange', action='store', default=2, type=int)
    parser.add_argument('-taug', action='store', default=False)
    parser.add_argument('-trange', action='store', default=2, type=int)
    parser.add_argument('-resample', action='store', default=True)
    parser.add_argument('-tresample', action='store', default=False)
    parser.add_argument('-hmatch', action='store', default=False)
    parser.add_argument('-gausweight', action='store', default=20, type=int)
    parser.add_argument('-gaussigma', action='store', default=1, type=int)

    results = parser.parse_args()
    print('given parameters: {}'.format(results))

    try:
        import distutils.util
        in_memory = distutils.util.strtobool(results.inmemory)
        if in_memory:
            print('running in-memory={}, watch for memory overflow!'.format(in_memory))
        #main(results, in_memory=in_memory)
    except Exception as e:
        print(e)
    main(results, in_memory=in_memory)
    exit()
