# from fuxictr.pytorch.models import BaseModel
import torch
from torch import nn
from fuxictr.pytorch import layers
import numpy as np
import pandas as pd
import logging
import random
from fuxictr.pytorch.torch_utils import get_initializer
from collections import OrderedDict
import os
# from fuxictr.pytorch.layers import LogisticRegression

import math
from tqdm import tqdm
import time
import sys
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import MLP_Block, CrossNet, FeatureTayEmbedding

from fuxictr.metrics import evaluate_metrics
from fuxictr.pytorch.torch_utils import get_device, get_optimizer, get_loss, get_regularizer, get_activation
from fuxictr.utils import Monitor
from itertools import product
import h5py

from model_zoo.DCN.DCN_torch.utils import feature_cross_importance


class LogisticRegression(nn.Module):
    def __init__(self, feature_map, use_bias=True, **kwargs):
        super(LogisticRegression, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True) if use_bias else None
        # A trick for quick one-hot encoding in LR
        self.embedding_layer = FeatureTayEmbedding(feature_map, 1, use_pretrain=False, use_sharing=False,
                                                   com_num=feature_map.com_num, paths=kwargs.get('paths'),
                                                   method=kwargs.get('method'), mode=kwargs.get('mode'))
    def forward(self, X):
        embed_weights = self.embedding_layer(X)
        output = embed_weights.sum(dim=1)
        if self.bias is not None:
            output += self.bias
        return output


class LR(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="LR", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 regularizer=None,
                 **kwargs):
        super(LR, self).__init__(feature_map, 
                                 model_id=model_id, 
                                 gpu=gpu, 
                                 embedding_regularizer=regularizer, 
                                 net_regularizer=regularizer, 
                                 **kwargs)
        self.lr_layer = LogisticRegression(feature_map, use_bias=True, paths=kwargs.get('paths'),
                                           method=kwargs.get('method'), mode=kwargs.get('mode'))
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X, y]
        """
        X = self.get_inputs(inputs)
        y_pred = self.lr_layer(X)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def evaluate_with_pfi(self, data_generator, valid_result = None, metrics=None, seed=2019):
        pfi_score_res = pd.DataFrame(columns=['AUC', 'logloss'])

        ## FIXME: 1 means the label column
        feature_com = self.lr_layer.embedding_layer.embedding_layer.feature_com_choice
        total_field = len(feature_com)
        for feat_idx in range(total_field):
            assert os.getenv('CUR_PERM_FEAT') is None, 'CUR_PERM_FEAT should be None'
            os.environ['CUR_PERM_FEAT'] = feature_com[feat_idx]
            cur_result = self.evaluate(data_generator, metrics=metrics)
            diff = pd.DataFrame([{
                'AUC': cur_result['AUC'] - valid_result['AUC'],
                'logloss': cur_result['logloss'] - valid_result['logloss']
            }])
            pfi_score_res = pd.concat([pfi_score_res, diff], ignore_index=True)

            os.environ.pop('CUR_PERM_FEAT', None)

        # Add feature name
        pfi_score_res.insert(0,'feature_name', feature_com)

        # Sort by AUC and see AUC as the feature_weight
        pfi_score_res.insert(1, 'feature_gain', pfi_score_res['logloss'])
        self.save_scores(pfi_score_res, 'feature_gain')
        return

    def _permute_feature(self,data_generator, feature_idx):
        """
        Permutes the values of a specific feature in each batch produced by the data_generator.

        Args:
        - data_generator: Original data generator.
        - feature_idx: The index of the feature you want to permute.

        Yields:
        - Batch with permuted feature values.
        """
        for batch in data_generator:
            # Deep copy to avoid modifying the original batch
            permuted_batch = batch.clone()

            # Permute the feature using PyTorch functions
            perm = torch.randperm(permuted_batch.size(0))
            permuted_batch[:, feature_idx] = permuted_batch[perm, feature_idx]

            yield permuted_batch


