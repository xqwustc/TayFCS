import torch
from torch import nn
import h5py
import os
import numpy as np
from collections import OrderedDict

from feat_select.AdaFS_module import controller_mlp
from fuxictr.pytorch.torch_utils import get_initializer
from fuxictr.pytorch import layers
import random
import pandas as pd
import logging


class FeatureTayEmbedding(nn.Module):
    def __init__(self,
                 feature_map,
                 embedding_dim,
                 embedding_initializer="partial(nn.init.normal_, std=1e-4)",
                 required_feature_columns=None,
                 not_required_feature_columns=None,
                 use_pretrain=True,
                 use_sharing=True,
                 com_num=0,
                 method='gncfs',
                 mode='weight',
                 paths=None):
        super(FeatureTayEmbedding, self).__init__()
        self.embedding_layer = FeatureEmbeddingDict(feature_map,
                                                    embedding_dim,
                                                    embedding_initializer=embedding_initializer,
                                                    required_feature_columns=required_feature_columns,
                                                    not_required_feature_columns=not_required_feature_columns,
                                                    use_pretrain=use_pretrain,
                                                    use_sharing=use_sharing,
                                                    com_num=com_num,
                                                    method=method,
                                                    mode=mode,
                                                    paths=paths)

    def forward(self, X, feature_source=[], feature_type=[], flatten_emb=False):
        feature_emb_dict = self.embedding_layer(X, feature_source=feature_source, feature_type=feature_type)
        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=flatten_emb)
        return feature_emb


class FeatureEmbeddingDict(nn.Module):
    def __init__(self,
                 feature_map,
                 embedding_dim,
                 embedding_initializer="partial(nn.init.normal_, std=1e-4)",
                 required_feature_columns=None,
                 not_required_feature_columns=None,
                 use_pretrain=True,
                 use_sharing=True,
                 com_num=0,
                 method='gncfs',
                 mode='weight',
                 paths=None):
        super(FeatureEmbeddingDict, self).__init__()
        self._feature_map = feature_map
        self.required_feature_columns = required_feature_columns
        self.not_required_feature_columns = not_required_feature_columns
        self.use_pretrain = use_pretrain
        self.embedding_initializer = embedding_initializer
        self.embedding_layers = nn.ModuleDict()
        self.feature_encoders = nn.ModuleDict()
        for feature, feature_spec in self._feature_map.features.items():
            if self.is_required(feature):
                if not (use_pretrain and use_sharing) and embedding_dim == 1:
                    feat_emb_dim = 1  # in case for LR
                    if feature_spec["type"] == "sequence":
                        self.feature_encoders[feature] = layers.MaskedSumPooling()
                else:
                    feat_emb_dim = feature_spec.get("embedding_dim", embedding_dim)
                    if feature_spec.get("feature_encoder", None):
                        self.feature_encoders[feature] = self.get_feature_encoder(feature_spec["feature_encoder"])

                # Set embedding_layer according to share_embedding
                if use_sharing and feature_spec.get("share_embedding") in self.embedding_layers:
                    self.embedding_layers[feature] = self.embedding_layers[feature_spec["share_embedding"]]
                    continue

                if feature_spec["type"] == "numeric":
                    self.embedding_layers[feature] = nn.Linear(1, feat_emb_dim, bias=False)
                elif feature_spec["type"] == "categorical":
                    padding_idx = feature_spec.get("padding_idx", None)
                    embedding_matrix = nn.Embedding(feature_spec["vocab_size"],
                                                    feat_emb_dim,
                                                    padding_idx=padding_idx)
                    if use_pretrain and "pretrained_emb" in feature_spec:
                        embedding_matrix = self.load_pretrained_embedding(embedding_matrix,
                                                                          feature_map,
                                                                          feature,
                                                                          freeze=feature_spec["freeze_emb"],
                                                                          padding_idx=padding_idx)
                    self.embedding_layers[feature] = embedding_matrix
                elif feature_spec["type"] == "sequence":
                    padding_idx = feature_spec.get("padding_idx", None)
                    embedding_matrix = nn.Embedding(feature_spec["vocab_size"],
                                                    feat_emb_dim,
                                                    padding_idx=padding_idx)
                    if use_pretrain and "pretrained_emb" in feature_spec:
                        embedding_matrix = self.load_pretrained_embedding(embedding_matrix,
                                                                          feature_map,
                                                                          feature,
                                                                          freeze=feature_spec["freeze_emb"],
                                                                          padding_idx=padding_idx)
                    self.embedding_layers[feature] = embedding_matrix
        # --- update for LiteFCS ---
        self.com_num = com_num
        if com_num:
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if type(paths) == dict:
                if paths.get('score_path'):
                    assert os.path.exists(paths.get('score_path')), f"score_path {paths.get('score_path')} does not exist"
            else:
                tpath = {}
                tpath['score_path'] = paths
                paths = tpath
            feature_com_importance = pd.read_csv(paths.get('score_path'))

            # log the top-k feature coms
            logging.info(f'Feature combinations importance: {feature_com_importance}')

            self.feature_com_choice = []

            feature_com_importance = feature_com_importance.sort_values(by='score', ascending=False)
            feature_com_name = feature_com_importance["combination"].tolist()

            upper_bound = 5000000  # for hash table size
            max_size = 5e20   # for feature used size
            if mode.endswith('hash'):
                upper_bound = 5e6
            self.com_vocabs = {}

            if mode.startswith('lre'):
                if feature_map.dataset_id == 'ipinyou_x1':
                    upper_bound = upper_bound // 5
                logging.info('Use LRE to select feature combinations')
                # eliminated features
                ele_path = paths.get('ele_path')
                assert os.path.exists(ele_path), f"ele_path {ele_path} does not exist"
                ele_feat = pd.read_csv(ele_path)

                # only get whose delta logloss < 0
                ele_feat = ele_feat[ele_feat['logloss'] < 0]


            for feature_com in feature_com_name:
                feature_son = feature_com.split(" & ")
                if len(feature_son) == 1:
                    # Do not add single feature
                    continue

                if 'ele_feat' in locals():
                    if feature_com in ele_feat['feature_name'].tolist():
                        continue
                if any(f in ['artist_name', 'genre_ids'] for f in feature_son):
                    continue

                tot_vocab_size = 1
                for cur_feature in feature_son:
                    tot_vocab_size *= self._feature_map.features[cur_feature]["vocab_size"]
                # if self._feature_map.features[feature_son[0]]["vocab_size"] * \
                #         self._feature_map.features[feature_son[1]]["vocab_size"] < 100000000:
                if max_size > tot_vocab_size > 0:
                    self.feature_com_choice.append(feature_com)
                    if len(self.feature_com_choice) == com_num:
                        break


            logging.info(f'Add new combinations:{self.feature_com_choice}')
            self.com_emb_dims = 0
            for feature_com in self.feature_com_choice:
                feature_son = feature_com.split(" & ")

                tot_vocab_size = 1
                for cur_feature in feature_son:
                    tot_vocab_size *= self._feature_map.features[cur_feature]["vocab_size"]

                if mode.endswith('hash'):
                    tot_vocab_size = min(tot_vocab_size, upper_bound) + 1
                self.com_vocabs[feature_com] = int(tot_vocab_size)

                cur_dim = embedding_dim

                embedding_matrix = nn.Embedding(self.com_vocabs[feature_com], cur_dim, padding_idx=padding_idx)
                logging.info(f'Add new com:{feature_com} with feat_emb_dim:{cur_dim} and '
                             f'tot_vocab_size:{self.com_vocabs[feature_com]}')
                self.com_emb_dims += cur_dim
                self.embedding_layers[feature_com] = embedding_matrix

        self.reset_parameters()

    def get_feature_encoder(self, encoder):
        try:
            if type(encoder) == list:
                encoder_list = []
                for enc in encoder:
                    encoder_list.append(eval(enc))
                encoder_layer = nn.Sequential(*encoder_list)
            else:
                encoder_layer = eval(encoder)
            return encoder_layer
        except:
            raise ValueError("feature_encoder={} is not supported.".format(encoder))

    def reset_parameters(self):
        self.embedding_initializer = get_initializer(self.embedding_initializer)
        for k, v in self.embedding_layers.items():
            if self.use_pretrain and k in self._feature_map.features and "pretrained_emb" in self._feature_map.features[
                k]:  # skip pretrained
                continue
            if k in self._feature_map.features and "share_embedding" in self._feature_map.features[
                k] and v.weight.requires_grad == False:
                continue
            if type(v) == nn.Embedding:
                if v.padding_idx is not None:  # using 0 index as padding_idx
                    self.embedding_initializer(v.weight[1:, :])
                else:
                    self.embedding_initializer(v.weight)

    def is_required(self, feature):
        """ Check whether feature is required for embedding """
        feature_spec = self._feature_map.features[feature]
        if feature_spec["type"] == "meta":
            return False
        elif self.required_feature_columns and (feature not in self.required_feature_columns):
            return False
        elif self.not_required_feature_columns and (feature in self.not_required_feature_columns):
            return False
        else:
            return True

    def get_pretrained_embedding(self, pretrained_path, feature_name):
        with h5py.File(pretrained_path, 'r') as hf:
            embeddings = hf[feature_name][:]
        return embeddings

    def load_pretrained_embedding(self, embedding_matrix, feature_map, feature_name, freeze=False, padding_idx=None):
        pretrained_path = os.path.join(feature_map.data_dir, feature_map.features[feature_name]["pretrained_emb"])
        embeddings = self.get_pretrained_embedding(pretrained_path, feature_name)
        if padding_idx is not None:
            embeddings[padding_idx] = np.zeros(embeddings.shape[-1])
        assert embeddings.shape[-1] == embedding_matrix.embedding_dim, \
            "{}\'s embedding_dim is not correctly set to match its pretrained_emb shape".format(feature_name)
        embeddings = torch.from_numpy(embeddings).float()
        embedding_matrix.weight = torch.nn.Parameter(embeddings)
        if freeze:
            embedding_matrix.weight.requires_grad = False
        return embedding_matrix

    def dict2tensor(self, embedding_dict, feature_list=[], feature_source=[], feature_type=[], flatten_emb=False):
        if type(feature_source) != list:
            feature_source = [feature_source]
        if type(feature_type) != list:
            feature_type = [feature_type]
        feature_emb_list = []
        for feature, feature_spec in self._feature_map.features.items():
            if feature_source and feature_spec["source"] not in feature_source:
                continue
            if feature_type and feature_spec["type"] not in feature_type:
                continue
            if feature_list and feature not in feature_list:
                continue
            if feature in embedding_dict:
                feature_emb_list.append(embedding_dict[feature])
        
        if self.com_num:
            for feature_com in self.feature_com_choice:
                feature_emb_list.append(embedding_dict[feature_com])
        
        if flatten_emb:
            feature_emb = torch.cat(feature_emb_list, dim=-1)
        else:
            feature_emb = torch.stack(feature_emb_list, dim=1)
        # print(feature_emb)
        # exit()
        return feature_emb

    def forward(self, inputs, feature_source=[], feature_type=[]):
        if type(feature_source) != list:
            feature_source = [feature_source]
        if type(feature_type) != list:
            feature_type = [feature_type]
        feature_emb_dict = OrderedDict()
        for feature, feature_spec in self._feature_map.features.items():
            if feature_source and feature_spec["source"] not in feature_source:
                continue
            if feature_type and feature_spec["type"] not in feature_type:
                continue
            if feature in self.embedding_layers:
                if feature_spec["type"] == "numeric":
                    # continue
                    inp = inputs[feature].float().view(-1, 1)
                    embeddings = self.embedding_layers[feature](inp)
                elif feature_spec["type"] == "categorical":
                    inp = inputs[feature].long()
                    embeddings = self.embedding_layers[feature](inp)
                elif feature_spec["type"] == "sequence":
                    inp = inputs[feature].long()
                    embeddings = self.embedding_layers[feature](inp)
                else:
                    raise NotImplementedError
                if feature in self.feature_encoders:
                    embeddings = self.feature_encoders[feature](embeddings)
                feature_emb_dict[feature] = embeddings
        
        if self.com_num:
            for feature_com in self.feature_com_choice:
                feature_son = feature_com.split(" & ")
                if len(feature_son) == 1:
                    continue

                # extend to higher order
                for idx, cur_feature in enumerate(feature_son):
                    if idx == 0:
                        input_com = inputs[cur_feature].long()
                    else:
                        input_com = input_com * self._feature_map.features[cur_feature]["vocab_size"] + \
                                      inputs[cur_feature].long()

                input_com = input_com.long()

                if hasattr(self, 'com_vocabs'):
                    vocab_size = self.com_vocabs[feature_com]
                    input_com = (input_com % (vocab_size - 1)) + 1

                embeddings = self.embedding_layers[feature_com](input_com)
                feature_emb_dict[feature_com] = embeddings

                if os.getenv('CUR_PERM_FEAT') is not None and feature_com == os.getenv('CUR_PERM_FEAT'):
                    feature_emb_dict[feature_com] = torch.mean(embeddings).unsqueeze(0).repeat(
                        embeddings.size(0), 1)
                    # feature_emb_dict[feature_com] = torch.zeros_like(feature_emb_dict[feature_com])
                    # logging.info(f'Permute features: {feature_com} done.')
                else:
                    feature_emb_dict[feature_com] = embeddings

        return feature_emb_dict