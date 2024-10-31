import json
import logging
import os
import shutil
import stat
from collections import OrderedDict
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
import requests
import numpy as np
from numpy import ndarray
import transformers
from huggingface_hub import HfApi, HfFolder, Repository, hf_hub_url, cached_download
import torch
from torch import nn, Tensor, device
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from tqdm.autonotebook import trange
import math
import queue
import tempfile
from distutils.dir_util import copy_tree
import copy

from sentence_transformers import __MODEL_HUB_ORGANIZATION__
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import import_from_string, batch_to_device, fullname, snapshot_download
from sentence_transformers.models import Transformer, Pooling, Dense
from sentence_transformers.model_card_templates import ModelCardTemplate
from sentence_transformers import __version__
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class CrossEntropySentenceTransformer(SentenceTransformer):
    """
    Loads or create a SentenceTransformer model, that can be used to map sentences / text to embeddings.

    :param model_name_or_path: If it is a filepath on disc, it loads the model from that path. If it is not a path, it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model from Huggingface models repository with that name.
    :param modules: This parameter can be used to create custom SentenceTransformer models from scratch.
    :param device: Device (like 'cuda' / 'cpu') that should be used for computation. If None, checks if a GPU can be used.
    :param cache_folder: Path to store models. Can be also set by SENTENCE_TRANSFORMERS_HOME enviroment variable.
    :param use_auth_token: HuggingFace authentication token to download private models.
    """

    def custom_smart_batching_collate(self, batch):
            """
            Note: nothing is changed from the default version yet, this is just a placeholder for future changes

            Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
            Here, batch is a list of tuples: [(tokens, label), ...]

            :param batch:
                a batch from a SmartBatchingDataset
            :return:
                a batch of tensors for the model
            """
            num_texts = len(batch[0].texts)
            texts = [[] for _ in range(num_texts)]
            labels = []

            for example in batch:
                for idx, text in enumerate(example.texts):
                    texts[idx].append(text) # for cross entropy loss there should only be 1 text per example wrapped in a list

                labels.append(example.label)

            labels = torch.tensor(labels)

            sentence_features = []
            for idx in range(num_texts):
                tokenized = self.tokenize(texts[idx])
                sentence_features.append(tokenized) # sentence_features[0] are the tokenized comments, it is a list of lists, inner list len=1
            
            return sentence_features, labels


    def CrossEntropyfit(self,
                train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
                classes: Tuple[str, ...],
                evaluator: SentenceEvaluator = None,
                epochs: int = 1,
                steps_per_epoch = None,
                scheduler: str = 'WarmupLinear',
                warmup_steps: int = 10000,
                optimizer_class: Type[Optimizer] = torch.optim.AdamW,
                optimizer_params : Dict[str, object]= {'lr': 2e-5},
                weight_decay: float = 0.01,
                evaluation_steps: int = 0,
                output_path: str = None,
                save_best_model: bool = True,
                max_grad_norm: float = 1,
                use_amp: bool = False,
                callback: Callable[[float, int, int], None] = None,
                show_progress_bar: bool = True,
                checkpoint_path: str = None,
                checkpoint_save_steps: int = 500,
                checkpoint_save_total_limit: int = 0
                ):
            """
            Train the model with the given training objective
            Each training objective is sampled in turn for one batch.
            We sample only as many batches from each objective as there are in the smallest one
            to make sure of equal training with each dataset.

            :param train_objectives: Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning
            :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
            :param epochs: Number of epochs for training
            :param steps_per_epoch: Number of training steps per epoch. If set to None (default), one epoch is equal the DataLoader size from train_objectives.
            :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
            :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
            :param optimizer_class: Optimizer
            :param optimizer_params: Optimizer parameters
            :param weight_decay: Weight decay for model parameters
            :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
            :param output_path: Storage path for the model and evaluation files
            :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
            :param max_grad_norm: Used for gradient normalization.
            :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
            :param callback: Callback function that is invoked after each evaluation.
                    It must accept the following three parameters in this order:
                    `score`, `epoch`, `steps`
            :param show_progress_bar: If True, output a tqdm progress bar
            :param checkpoint_path: Folder to save checkpoints during training
            :param checkpoint_save_steps: Will save a checkpoint after so many steps
            :param checkpoint_save_total_limit: Total number of checkpoints to store
            """

            ##Add info to model card
            #info_loss_functions = "\n".join(["- {} with {} training examples".format(str(loss), len(dataloader)) for dataloader, loss in train_objectives])
            info_loss_functions =  []
            for dataloader, loss in train_objectives:
                info_loss_functions.extend(ModelCardTemplate.get_train_objective_info(dataloader, loss))
            info_loss_functions = "\n\n".join([text for text in info_loss_functions])

            info_fit_parameters = json.dumps({"evaluator": fullname(evaluator), "epochs": epochs, "steps_per_epoch": steps_per_epoch, "scheduler": scheduler, "warmup_steps": warmup_steps, "optimizer_class": str(optimizer_class),  "optimizer_params": optimizer_params, "weight_decay": weight_decay, "evaluation_steps": evaluation_steps, "max_grad_norm": max_grad_norm }, indent=4, sort_keys=True)
            self._model_card_text = None
            self._model_card_vars['{TRAINING_SECTION}'] = ModelCardTemplate.__TRAINING_SECTION__.replace("{LOSS_FUNCTIONS}", info_loss_functions).replace("{FIT_PARAMETERS}", info_fit_parameters)


            if use_amp:
                from torch.cuda.amp import autocast
                scaler = torch.cuda.amp.GradScaler()

            self.to(self.device)

            dataloaders = [dataloader for dataloader, _ in train_objectives]

            # Use smart batching
            for dataloader in dataloaders:
                dataloader.collate_fn = self.custom_smart_batching_collate

            loss_models = [loss for _, loss in train_objectives]
            for loss_model in loss_models:
                loss_model.to(self.device)

            self.best_score = -9999999

            if steps_per_epoch is None or steps_per_epoch == 0:
                steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

            num_train_steps = int(steps_per_epoch * epochs)

            # Prepare optimizers
            optimizers = []
            schedulers = []
            for loss_model in loss_models:
                param_optimizer = list(loss_model.named_parameters())

                no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]

                optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
                scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

                optimizers.append(optimizer)
                schedulers.append(scheduler_obj)


            global_step = 0
            data_iterators = [iter(dataloader) for dataloader in dataloaders]

            num_train_objectives = len(train_objectives)

            skip_scheduler = False
            for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
                training_steps = 0

                for loss_model in loss_models:
                    loss_model.zero_grad()
                    loss_model.train()

                for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                    for train_idx in range(num_train_objectives):
                        loss_model = loss_models[train_idx]
                        optimizer = optimizers[train_idx]
                        scheduler = schedulers[train_idx]
                        data_iterator = data_iterators[train_idx]

                        try:
                            data = next(data_iterator)
                        except StopIteration:
                            data_iterator = iter(dataloaders[train_idx])
                            data_iterators[train_idx] = data_iterator
                            data = next(data_iterator)

                        features, labels = data # unpacked possible due to the use of smart_batching_collate
                        labels = labels.to(self.device)
                        features = list(map(lambda batch: batch_to_device(batch, self.device), features))
                        classes_copy = list(copy.deepcopy(classes))
                        classes_copy = self.tokenize(classes_copy)
                        classes_copy = {key: value.clone().detach().to(self.device) for key, value in classes_copy.items()}
                 

                        if use_amp:
                            with autocast():
                                loss_value = loss_model(features, labels, classes_copy)

                            scale_before_step = scaler.get_scale()
                            scaler.scale(loss_value).backward()
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                            scaler.step(optimizer)
                            scaler.update()

                            skip_scheduler = scaler.get_scale() != scale_before_step
                        else:
                            loss_value = loss_model(features, labels, classes_copy)
                            loss_value.backward()
                            torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                            optimizer.step()

                        optimizer.zero_grad()

                        if not skip_scheduler:
                            scheduler.step()

                    training_steps += 1
                    global_step += 1

                    if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                        self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)

                        for loss_model in loss_models:
                            loss_model.zero_grad()
                            loss_model.train()

                    if checkpoint_path is not None and checkpoint_save_steps is not None and checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0:
                        self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)


                self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)

            if evaluator is None and output_path is not None:   #No evaluator, but output path: save final model version
                self.save(output_path)

            if checkpoint_path is not None:
                self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)


