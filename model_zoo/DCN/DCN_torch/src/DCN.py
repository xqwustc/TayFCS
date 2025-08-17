import tqdm
import sys
import numpy as np
import pandas as pd
import logging
import h5py
import os
from collections import OrderedDict

from tqdm import tqdm
import random
import time

import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
from fuxictr.pytorch.layers import MLP_Block, CrossNet
from fuxictr.pytorch.layers import FeatureTayEmbedding
from fuxictr.pytorch.models import BaseModel

from itertools import product

class DCN(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="DCN",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 dnn_hidden_units=[],
                 dnn_activations="ReLU",
                 num_cross_layers=3,
                 net_dropout=0.3,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 com_num=0,
                 use_mlp=False,
                 **kwargs):
        super(DCN, self).__init__(feature_map,
                                  model_id=model_id,
                                  gpu=gpu,
                                  embedding_regularizer=embedding_regularizer,
                                  net_regularizer=net_regularizer,
                                  **kwargs)

        self.mode = kwargs.get('mode', None)
        self.method = kwargs.get('method', None)

        # --- update for tayfcs retrain start ---
        paths = kwargs.get('paths')
        if paths:
            paths['score_path'] = self.read_scores(score_version=paths.get('score_path', None), path_only=True)
            paths['ele_path'] = self.read_scores(score_name='feature_gain', score_version=paths.get('ele_path', None),
                                                 path_only=True)
            logging.info(f"Score path: {paths['score_path']} and elimination path: {paths['ele_path']}")
        if self.method == 'tayfcs' or self.method.startswith('re'):
            self.embedding_layer = FeatureTayEmbedding(feature_map, embedding_dim,
                                                       com_num=com_num, method=self.method,
                                                       mode=self.mode, paths=paths)
        else:
            self.embedding_layer = FeatureTayEmbedding(feature_map, embedding_dim)

        input_dim = feature_map.sum_emb_out_dim()
        self.embed_dim = embedding_dim

        if com_num:
            input_dim += self.embedding_layer.embedding_layer.com_emb_dims

        self.dnn = MLP_Block(input_dim=input_dim,
                             output_dim=None,  # output hidden layer
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm) if dnn_hidden_units else None

        self.crossnet = CrossNet(input_dim, num_cross_layers)
        final_dim = input_dim
        if isinstance(dnn_hidden_units, list) and len(dnn_hidden_units) > 0:  # if use dnn
            final_dim += dnn_hidden_units[-1]
        # --- update for mlp ---
        self.use_mlp = use_mlp
        if self.use_mlp:
            final_dim = dnn_hidden_units[-1] + embedding_dim * com_num
        # --- update for mlp ---

        self.fc = nn.Linear(final_dim, 1)  # [cross_part, dnn_part] -> logit

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, flatten_emb=True)

        cross_out = self.crossnet(feature_emb)
        if self.dnn is not None:
            dnn_out = self.dnn(feature_emb)
            final_out = torch.cat([cross_out, dnn_out], dim=-1)
            # --- update for mlp ---
            if self.use_mlp:
                final_out = dnn_out
            # --- update for mlp ---
        else:
            final_out = cross_out
        y_pred = self.fc(final_out)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict