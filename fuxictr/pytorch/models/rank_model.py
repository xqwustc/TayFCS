# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import torch.nn as nn
import numpy as np
import torch
import os, sys
import logging
from fuxictr.metrics import evaluate_metrics
from fuxictr.pytorch.torch_utils import get_device, get_optimizer, get_loss, get_regularizer
from fuxictr.utils import Monitor
from tqdm import tqdm
import pandas as pd
import time
from torch.optim.lr_scheduler import LambdaLR
from sklearn.decomposition import PCA

from tests.unit_tests.test_pytorch_layers_bilinear_interaction import embedding_dim


class BaseModel(nn.Module):
    def __init__(self, 
                 feature_map, 
                 model_id="BaseModel", 
                 task="binary_classification", 
                 gpu=-1, 
                 monitor="AUC", 
                 save_best_only=True, 
                 monitor_mode="max", 
                 early_stop_patience=2, 
                 eval_steps=None, 
                 embedding_regularizer=None, 
                 net_regularizer=None, 
                 reduce_lr_on_plateau=True,
                 **kwargs):
        super(BaseModel, self).__init__()
        self.device = get_device(gpu)
        self._monitor = Monitor(kv=monitor)
        self._monitor_mode = monitor_mode
        self._early_stop_patience = early_stop_patience
        self._eval_steps = eval_steps # None default, that is evaluating every epoch
        self._save_best_only = save_best_only
        self._embedding_regularizer = embedding_regularizer
        self._net_regularizer = net_regularizer
        self._reduce_lr_on_plateau = reduce_lr_on_plateau
        self._verbose = kwargs["verbose"]
        self.feature_map = feature_map
        self.output_activation = self.get_output_activation(task)
        self.model_id = model_id
        self.weight_decay = kwargs.get("weight_decay", None)
        self.model_dir = os.path.join(kwargs["model_root"], feature_map.dataset_id)
        self.checkpoint = os.path.abspath(os.path.join(self.model_dir, self.model_id + ".model"))
        self.validation_metrics = kwargs["metrics"]

    def compile(self, optimizer, loss, lr, scheduler=None, warmup_steps=500, total_steps=10000, gamma = 0.95):
        self.optimizer = get_optimizer(optimizer, self.parameters(), lr, self.weight_decay)
        if scheduler is not None:
            if scheduler.lower() == "cosine":
                # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)
                self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda(warmup_steps, total_steps))
                logging.info(f"Use CosineAnnealingLR scheduler {self.scheduler}.")
            elif scheduler == "exp":  # exponential decay
                self.scheduler = LambdaLR(self.optimizer,
                                          lr_lambda=self.exp_lr_lambda(warmup_steps, total_steps, gamma))
                logging.info(f"Use Exponential Decay scheduler {self.scheduler}.")
        self.loss_fn = get_loss(loss)

    def lr_lambda(self, warmup_steps, total_steps):
        def lambda_lr(step):
            if step < warmup_steps:
                return float(step) / warmup_steps  # Warm-up
            return 0.5 * (1 + torch.cos(torch.tensor(step / total_steps * torch.pi)))  # Cosine Annealing 衰减策略
        return lambda_lr

    def exp_lr_lambda(self, warmup_steps, total_steps, gamma):
        def lambda_lr(step):
            if step < warmup_steps:
                return float(step) / warmup_steps  # linear warmup
            return gamma ** ((step - warmup_steps) / (total_steps - warmup_steps))  # exponential decay

        return lambda_lr

    def regularization_loss(self):
        reg_term = 0
        if self._embedding_regularizer or self._net_regularizer:
            emb_reg = get_regularizer(self._embedding_regularizer)
            net_reg = get_regularizer(self._net_regularizer)
            for _, module in self.named_modules():
                for p_name, param in module.named_parameters():
                    if param.requires_grad:
                        if p_name in ["weight", "bias"]:
                            if type(module) == nn.Embedding:
                                if self._embedding_regularizer:
                                    for emb_p, emb_lambda in emb_reg:
                                        reg_term += (emb_lambda / emb_p) * torch.norm(param, emb_p) ** emb_p
                            else:
                                if self._net_regularizer:
                                    for net_p, net_lambda in net_reg:
                                        reg_term += (net_lambda / net_p) * torch.norm(param, net_p) ** net_p
        return reg_term

    def compute_loss(self, return_dict, y_true):
        loss = self.loss_fn(return_dict["y_pred"], y_true, reduction='mean')
        loss += self.regularization_loss()
        return loss

    def reset_parameters(self):
        def reset_default_params(m):
            # initialize nn.Linear/nn.Conv1d layers by default
            if type(m) in [nn.Linear, nn.Conv1d]:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        def reset_custom_params(m):
            # initialize layers with customized reset_parameters
            if hasattr(m, 'reset_custom_params'):
                m.reset_custom_params()
        self.apply(reset_default_params)
        self.apply(reset_custom_params)

    def get_inputs(self, inputs, feature_source=None):
        if feature_source and type(feature_source) == str:
            feature_source = [feature_source]
        X_dict = dict()
        for feature, spec in self.feature_map.features.items():
            if (feature_source is not None) and (spec["source"] not in feature_source):
                continue
            if spec["type"] == "meta":
                continue
            X_dict[feature] = inputs[:, self.feature_map.get_column_index(feature)].to(self.device)
        return X_dict

    def get_labels(self, inputs):
        """ Please override get_labels() when using multiple labels!
        """
        labels = self.feature_map.labels
        y = inputs[:, self.feature_map.get_column_index(labels[0])].to(self.device)
        return y.float().view(-1, 1)
                
    def get_group_id(self, inputs):
        return inputs[:, self.feature_map.get_column_index(self.feature_map.group_id)]

    def model_to_device(self):
        self.to(device=self.device)

    def lr_decay(self, factor=0.1, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            reduced_lr = max(param_group["lr"] * factor, min_lr)
            param_group["lr"] = reduced_lr
        return reduced_lr
           
    def fit(self, data_generator, epochs=1, validation_data=None,
            max_gradient_norm=10., **kwargs):
        self.valid_gen = validation_data
        self.test_gen = kwargs.get("test_data")
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.Inf if self._monitor_mode == "min" else -np.Inf
        self._stopping_steps = 0
        self._steps_per_epoch = len(data_generator)
        self._stop_training = False
        self._total_steps = 0
        self._batch_index = 0
        self._epoch_index = 0
        if self._eval_steps is None:
            self._eval_steps = self._steps_per_epoch
        
        logging.info("Start training: {} batches/epoch".format(self._steps_per_epoch))
        logging.info("************ Epoch=1 start ************")
        for epoch in range(epochs):
            self._epoch_index = epoch
            self.train_epoch(data_generator)
            if self._stop_training:
                break
            else:
                logging.info("************ Epoch={} end ************".format(self._epoch_index + 1))
        logging.info("Training finished.")
        logging.info("Load best model: {}".format(self.checkpoint))
        self.load_weights(self.checkpoint)

    def checkpoint_and_earlystop(self, logs, min_delta=1e-6):
        monitor_value = self._monitor.get_value(logs)
        if (self._monitor_mode == "min" and monitor_value > self._best_metric - min_delta) or \
           (self._monitor_mode == "max" and monitor_value < self._best_metric + min_delta):
            self._stopping_steps += 1
            logging.info("Monitor({})={:.6f} STOP!".format(self._monitor_mode, monitor_value))
            if self._reduce_lr_on_plateau:
                current_lr = self.lr_decay()
                logging.info("Reduce learning rate on plateau: {:.6f}".format(current_lr))
        else:
            self._stopping_steps = 0
            self._best_metric = monitor_value
            if self._save_best_only:
                logging.info("Save best model: monitor({})={:.6f}"\
                             .format(self._monitor_mode, monitor_value))
                self.save_weights(self.checkpoint)
        if self._stopping_steps >= self._early_stop_patience:
            self._stop_training = True
            logging.info("********* Epoch=={} early stop *********".format(self._epoch_index + 1))
        if not self._save_best_only:
            self.save_weights(self.checkpoint)

    def eval_step(self):
        logging.info('Evaluation @epoch {} - batch {}: '.format(self._epoch_index + 1, self._batch_index + 1))
        # if self.test_gen is not None:
        #     test_logs = self.evaluate(self.test_gen, metrics=self._monitor.get_metrics())
        val_logs = self.evaluate(self.valid_gen, metrics=self._monitor.get_metrics())
        self.checkpoint_and_earlystop(val_logs)
        self.train()

    def train_step(self, batch_data):
        self.optimizer.zero_grad()
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss = self.compute_loss(return_dict, y_true)
        loss.backward()
        # nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()
        if hasattr(self, "scheduler"):
            self.scheduler.step()
        return loss

    def train_epoch(self, data_generator):
        self._batch_index = 0
        train_loss = 0
        self.train()
        if self._verbose == 0:
            batch_iterator = data_generator
        else:
            batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
        for batch_index, batch_data in enumerate(batch_iterator):
            self._batch_index = batch_index
            self._total_steps += 1
            loss = self.train_step(batch_data)
            train_loss += loss.item()
            if self._total_steps % self._eval_steps == 0:
                time.sleep(2)
                logging.info("Train loss: {:.6f}".format(train_loss / self._eval_steps))
                logging.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
                train_loss = 0
                self.eval_step()
            if self._stop_training:
                break

    def evaluate(self, data_generator, metrics=None):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            y_true = []
            group_id = []
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                y_true.extend(self.get_labels(batch_data).data.cpu().numpy().reshape(-1))
                if self.feature_map.group_id is not None:
                    group_id.extend(self.get_group_id(batch_data).numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)
            group_id = np.array(group_id) if len(group_id) > 0 else None
            if metrics is not None:
                val_logs = self.evaluate_metrics(y_true, y_pred, metrics, group_id)
            else:
                val_logs = self.evaluate_metrics(y_true, y_pred, self.validation_metrics, group_id)
            logging.info('[Metrics] ' + ' - '.join('{}: {:.6f}'.format(k, v) for k, v in val_logs.items()))
            return val_logs

    def predict(self, data_generator):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            if self._verbose > 0:
                data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            return y_pred

    def evaluate_metrics(self, y_true, y_pred, metrics, group_id=None):
        return evaluate_metrics(y_true, y_pred, metrics, group_id)

    def save_weights(self, checkpoint):
        torch.save(self.state_dict(), checkpoint)
    
    def load_weights(self, checkpoint):
        self.to(self.device)
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.load_state_dict(state_dict)

    def get_output_activation(self, task):
        if task == "binary_classification":
            return nn.Sigmoid()
        elif task == "regression":
            return nn.Identity()
        else:
            raise NotImplementedError("task={} is not supported.".format(task))

    def count_parameters(self, count_embedding=True):
        total_params = 0
        for name, param in self.named_parameters(): 
            if not count_embedding and "embedding" in name:
                continue
            if param.requires_grad:
                total_params += param.numel()
        logging.info("Total number of parameters: {}.".format(total_params))

    def evaluate_with_tayscorer(self, data_generator, use_novelty=True):
        score_final_result = dict()
        for batch_data in tqdm(data_generator):
            X = self.get_inputs(batch_data)
            self.y_true = self.get_labels(batch_data)
            self.feature_emb = self.embedding_layer(X, flatten_emb=True)
            self.feature_emb.requires_grad_(requires_grad=True)
            self.feature_emb_mean = torch.mean(self.feature_emb, axis=0)
            self.feature_emb_delta = self.feature_emb - self.feature_emb_mean
            self.feature_emb.retain_grad()

            loss = self.forward_with_allgrads(self.feature_emb)
            loss.backward()

            embed_gradient = self.feature_emb.grad.unsqueeze(1) # [batch, 1, emb]
            embed_gradient_T = embed_gradient.permute(0, 2, 1) # [batch, emb, 1]

            taylor_delta = self.feature_emb_delta.unsqueeze(1)
            taylor_delta_T = taylor_delta.permute(0, 2, 1)

            second_grad = torch.bmm(embed_gradient_T, embed_gradient).detach()
            second_delta = torch.bmm(taylor_delta_T, taylor_delta).detach()
            second_score = second_grad * second_delta
            second_score = torch.abs(second_score)
            second_score = torch.mean(second_score, dim=0)

            third_grad = second_grad.unsqueeze(3) * embed_gradient.unsqueeze(1).detach()
            third_delta = second_delta.unsqueeze(3) * taylor_delta.unsqueeze(1).detach()
            third_score = third_grad * third_delta
            third_score = torch.abs(third_score)
            third_score = torch.mean(third_score, dim=0)


            for i in range(second_score.shape[0]):
                for j in range(second_score.shape[1]):
                    inx_list = [i, j]
                    # sort the list to get the unique key
                    inx_list = sorted(set(inx_list))
                    inx_tuple = tuple(inx_list)
                    if len(inx_tuple) > 1:
                        if inx_tuple not in score_final_result:
                            score_final_result[inx_tuple] = second_score[i, j].detach().cpu().numpy()
                        else:
                            score_final_result[inx_tuple] += second_score[i, j].detach().cpu().numpy()
                    for k in range(third_score.shape[2]):
                        inx_list = sorted(set([i, j, k]))
                        inx_list.sort()
                        inx_tuple = tuple(inx_list)
                        if len(inx_tuple) > 1:
                            if inx_tuple not in score_final_result:
                                score_final_result[inx_tuple] = third_score[i, j, k].detach().cpu().numpy()
                            else:
                                score_final_result[inx_tuple] += third_score[i, j, k].detach().cpu().numpy()
                        # for l in range(fourth_score.shape[3]):
                        #     inx_list = sorted(set([i, j, k, l]))
                        #     inx_list.sort()
                        #     inx_tuple = tuple(inx_list)
                        #     if len(inx_tuple) > 1:
                        #         if inx_tuple not in score_final_result:
                        #             score_final_result[inx_tuple] = fourth_score[i, j, k, l].detach().cpu().numpy()
                        #         else:
                        #             score_final_result[inx_tuple] += fourth_score[i, j, k, l].detach().cpu().numpy()

            # traverse
            self.optimizer.zero_grad()

        # sort the result by score
        score_final_result = pd.DataFrame(score_final_result.items(), columns=['combination', 'score'])
        score_final_result = score_final_result.sort_values(by='score', ascending=False)

        # replace the combination with feature names, combination is a tuple, tuple to a & b & c
        feature_name = list(self.feature_map.features.keys())
        for idx, row in score_final_result.iterrows():
            combination = row['combination']
            feature_name_combination = ' & '.join([feature_name[inx] for inx in combination])
            score_final_result.loc[idx, 'combination'] = feature_name_combination

        # save to csv
        self.save_scores(score_final_result)
        return

    def evaluate_with_tayscorer_native(self, data_generator, use_novelty=True):
        self.eval()
        self.evaluate_with_tayscorer(data_generator, use_novelty)
        return


    def save_scores(self, score_df, score_name="combine_score"):
        base_name = os.path.join(os.path.dirname(self.checkpoint), score_name)
        filename = f"{base_name}.csv"
        i = 1
        while os.path.exists(filename):
            filename = f"{base_name}_{i}.csv"
            i += 1

        score_df.to_csv(filename, index=False)
        logging.info(f"Feature scores saved to {filename}.")

    def read_scores(self, score_name="combine_score", score_version = "", ratio = 1, path_only = False):
        if score_version is None:
            return None
        base_name = os.path.join(os.path.dirname(self.checkpoint), score_name)

        if score_version != "":
            score_version = f"_{score_version}"

        filename = f"{base_name}{score_version}.csv"

        if not os.path.exists(filename):
            logging.warning(f"Feature scores not found: {filename}.")
            raise FileNotFoundError(f"Feature scores not found: {filename}.")
        else:
            if path_only:
                return filename
