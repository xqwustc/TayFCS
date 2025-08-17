import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys

sys.path.append('../../..')
sys.path.append('../../../../')

# --- update for fmcr ---
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern
import pandas as pd
# --- update for fmcr ---

import logging
import pickle
import copy
import numpy as np
from fuxictr import datasets
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders import H5DataLoader
from fuxictr.preprocess import FeatureProcessor, build_dataset
import src as model_zoo
import gc
import argparse
import os
import importlib

if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='DeepFM_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    parser.add_argument('--com_num', nargs='+', type=int, default=[10]) # 10,20])

    args = vars(parser.parse_args())

    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    # email.common_send('DCN_incre.py',params["data_format"])

    if params.get('spe_processor'):
        module_name = f"fuxictr.datasets.{params['spe_processor']}"
        fp_module = importlib.import_module(module_name)
        assert hasattr(fp_module, 'FeatureProcessor')
        FeatureProcessor = getattr(fp_module, 'FeatureProcessor')
    else:
        from fuxictr.preprocess import FeatureProcessor

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    if params["data_format"] == "csv":
        # Build feature_map and transform h5 data
        feature_encoder = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = \
            build_dataset(feature_encoder, **params)

    # here specify feature numbers
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))

    topk_column_name = []
    topk_logloss_result = []
    topk_auc_result = []
    time_consumption = []
    inf_times = []

    for i in args['com_num']:
        seed_everything(seed=params['seed'])
        topk_params = copy.deepcopy(params)
        topk_params['com_num'] = i
        logging.info('--- Use {} feature combinations !'.format(i))
        topk_feature_map = FeatureMap(topk_params['dataset_id'], data_dir)
        topk_feature_map.load(feature_map_json, topk_params)

        model_class = getattr(model_zoo, topk_params['model'])
        topk_model = model_class(topk_feature_map, **topk_params)
        topk_model.count_parameters()  # print number of parameters used in model

        train_gen,valid_gen,test_gen = H5DataLoader(topk_feature_map, stage='both', **topk_params).make_iterator()

        topk_model.fit(train_gen, validation_data=valid_gen, **params)

        valid_result = topk_model.evaluate(test_gen)

        topk_column_name.append(i)
        topk_logloss_result.append(valid_result['logloss'])
        topk_auc_result.append(valid_result['AUC'])
        cur_AUC = valid_result['AUC']

        del train_gen, valid_gen, test_gen
        gc.collect()


    topk_res = pd.DataFrame({'feature_name': topk_column_name,
                             'topk_logloss': topk_logloss_result,
                             'topk_auc': topk_auc_result})

    logging.info(topk_res)
    print(topk_res)