import glob
import numpy as np
import os
import pandas as pd
def evaluate_supervised(exp_root, result_df='results.xlsx', pred_suffix ='pred'):
    """
    Evaluate a cross-validation
    Expect to have predicted numpy files within each fold sub-dir
    Parameters
    ----------
    exp_root : (string) path to one experiment root, above the fold_n sub-folders
    result_df : (string) name of the results excel filename
    pred_suffix :

    Returns pd.DataFrame with the pFD per phase and patient
    -------

    """
    print('eval: {}'.format(exp_root))
    from src.utils.Metrics import meandiff
    phases = ['ED', 'MS', 'ES', 'PF', 'MD']

    pred_wildcard = '{}/{}/gtpred*.npy'.format(exp_root, pred_suffix)
    print('path to predictions: {}'.format(pred_wildcard))
    all_pred_files = glob.glob(pred_wildcard)
    assert len(all_pred_files) >0, 'we expect any predicted files, but found: {} predictions'.format(len(all_pred_files))
    print('predictions found: {}'.format(len(all_pred_files)))

    # Load the numpy prediction files
    preds = list(map(lambda x : np.load(x), all_pred_files))
    # stack the numpy files
    preds = np.concatenate(preds, axis=1)

    # calculate the mean differences
    res = meandiff(preds[0], preds[1], apply_sum=False, apply_average=False)
    df = pd.DataFrame(res.numpy(), columns=phases)
    df.to_excel(os.path.join(exp_root, result_df))
    return df

if __name__ == "__main__":
    import argparse, os, sys, glob

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    parser = argparse.ArgumentParser(description='predict a phase registration model')

    # usually the exp root parameters should yield to a config, which encapsulate all experiment parameters
    parser.add_argument('-exp_root', action='store', default='tmp')

    results = parser.parse_args()
    sys.path.append(os.getcwd())
    print('given parameters: {}'.format(results))

    # get all cfgs - we expect to find 4 as we usually train a 4-fold cv
    # call the predict_fn for each cfg
    initial_search_pattern = 'config/config.json' # path to one experiment
    search_path = os.path.join(results.exp_root, initial_search_pattern)
    cfg_files = sorted(glob.glob(search_path))

    try:
        evaluate_supervised(exp_root=results.exp_root)
    except Exception as e:
        print(e)
    exit()