import torch
from torch import nn
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)
from fuxictr.pytorch.layers import FeatureTayEmbedding, MLP_Block
from fuxictr.pytorch.models import BaseModel
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
from itertools import cycle
import itertools

class DNN(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="DNN", 
                 gpu=-1, 
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 hidden_units=[64, 64, 64], 
                 hidden_activations="ReLU", 
                 net_dropout=0, 
                 batch_norm=False, 
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 cross_num=0,
                 # method='gncfs',
                 # mode='weight',
                 **kwargs):
        super(DNN, self).__init__(feature_map, 
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
            paths['score_path'] = self.read_scores(score_version=paths.get('score_path',None), path_only=True)
            paths['ele_path'] = self.read_scores(score_name='feature_gain', score_version=paths.get('ele_path',None), path_only=True)
            logging.info(f"Score path: {paths['score_path']} and elimination path: {paths['ele_path']}")
        if not hasattr(self, "method"):
            self.embedding_layer = FeatureTayEmbedding(feature_map, embedding_dim)
        elif self.method == 'tayfcs' or self.method.startswith('re'):
            self.embedding_layer = FeatureTayEmbedding(feature_map, embedding_dim,
                                                       com_num=cross_num, method=self.method,
                                                       mode=self.mode, paths=paths)


        input_dim = feature_map.sum_emb_out_dim()
        self.embed_dim = embedding_dim

        if cross_num:
            input_dim += self.embedding_layer.embedding_layer.com_emb_dims

        self.mlp = MLP_Block(input_dim=input_dim,
                             output_dim=1, 
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=self.output_activation,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate, scheduler=kwargs.get("scheduler", None))
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        feature_emb = feature_emb.flatten(start_dim=1)
        y_pred = self.mlp(feature_emb)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def forward_with_allgrads(self, inputs):
        feature_emb = inputs

        y_pred = self.mlp(feature_emb)

        loss = self.compute_loss({"y_pred": y_pred}, self.y_true)
        return loss