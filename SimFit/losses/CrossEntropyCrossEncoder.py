from sentence_transformers.cross_encoder import CrossEncoder
import logging
import os
from typing import Callable
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from transformers import is_torch_npu_available
from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.readers import InputExample
from transformers.tokenization_utils_base import BatchEncoding
from typing import Optional, Tuple, List, Dict, Type
import copy
import random

logger = logging.getLogger(__name__)

class CrossEntropyCrossEncoder(CrossEncoder):
    def __init__(self, *args, classes:Optional[Tuple[str]]=None, subsample:int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.classes = classes
        self.subsample = subsample

        # Raise an error if subsample is larger than the number of classes
        if classes is not None and subsample > len(classes):
            raise ValueError(f"Subsample size ({subsample}) cannot be larger than the number of classes ({len(classes)}).")
     

    def cross_entropy_smart_batching_collate(self, batch: List[InputExample]) -> Tuple[BatchEncoding, Tensor]:
        tokenized_list = []
        labels = []

        classes_copy = list(copy.deepcopy(self.classes))
        classes_list = [class_copy.strip() for class_copy in classes_copy]

        # create a list of dictionaries, each dictionary contains an example paired with every class
        for example in batch:
            texts = [[],[]]
            text = example.texts[0].strip() # there should only be 1 text per example within the list
            filtered_list = classes_list[:example.label] + classes_list[example.label+1:]
            random_selection = random.sample(filtered_list, self.subsample-1)
            random_selection.append(classes_list[example.label]) # last item on the list would be the actual class
            for selection in random_selection:
                texts[0].append(text)
                texts[1].append(selection)
            tokenized = self.tokenizer(*texts, padding=True, truncation="longest_first", return_tensors="pt", max_length=self.max_length) 
            # tokenized has dimension within each key the number of classes (e.g. 83)
            for name in tokenized:
                tokenized[name] = tokenized[name].to(self._target_device)
            tokenized_list.append(tokenized) # this list has dimension equal to the batch size
            labels.append(self.subsample-1) # the last item on the list is always the actual class

        labels = torch.tensor(labels, dtype=torch.long).to(self._target_device)

        return tokenized_list, labels
    

    def CrossEntropyfit(
        self,
        train_dataloader: DataLoader,
        evaluator: SentenceEvaluator = None,
        epochs: int = 1,
        loss_fct=None,
        activation_fct=nn.Identity(),
        scheduler: str = "WarmupLinear",
        warmup_steps: int = 10000,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params: Dict[str, object] = {"lr": 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 0,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
    ) -> None:
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        Args:
            train_dataloader (DataLoader): DataLoader with training InputExamples
            evaluator (SentenceEvaluator, optional): An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc. Defaults to None.
            epochs (int, optional): Number of epochs for training. Defaults to 1.
            loss_fct: Which loss function to use for training. If None, will use nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss(). Defaults to None.
            activation_fct: Activation function applied on top of logits output of model.
            scheduler (str, optional): Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts. Defaults to "WarmupLinear".
            warmup_steps (int, optional): Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero. Defaults to 10000.
            optimizer_class (Type[Optimizer], optional): Optimizer. Defaults to torch.optim.AdamW.
            optimizer_params (Dict[str, object], optional): Optimizer parameters. Defaults to {"lr": 2e-5}.
            weight_decay (float, optional): Weight decay for model parameters. Defaults to 0.01.
            evaluation_steps (int, optional): If > 0, evaluate the model using evaluator after each number of training steps. Defaults to 0.
            output_path (str, optional): Storage path for the model and evaluation files. Defaults to None.
            save_best_model (bool, optional): If true, the best model (according to evaluator) is stored at output_path. Defaults to True.
            max_grad_norm (float, optional): Used for gradient normalization. Defaults to 1.
            use_amp (bool, optional): Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0. Defaults to False.
            callback (Callable[[float, int, int], None], optional): Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`. Defaults to None.
            show_progress_bar (bool, optional): If True, output a tqdm progress bar. Defaults to True.
        """
        train_dataloader.collate_fn = self.cross_entropy_smart_batching_collate

        if use_amp:
            if is_torch_npu_available():
                scaler = torch.npu.amp.GradScaler()
            else:
                scaler = torch.cuda.amp.GradScaler()
        self.model.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(
                optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps
            )

        if loss_fct is None:
            loss_fct = nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            for features, labels in tqdm(
                train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar
            ):
                if use_amp:
                    with torch.autocast(device_type=self._target_device.type):
                        loss_value = torch.tensor(0.0, device=self._target_device)
                        num_features = len(features) # batch size
                        """batch_features = {key: torch.cat([feature[key] for feature in features], dim=0) for key in features[0]}
                        model_predictions = self.model(**batch_features, return_dict=True)
                        # Apply the activation function to the logits
                        logits = activation_fct(model_predictions.logits)
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                        partitioned_losses = []
                        for i in range(num_features):
                            partition_logits = logits[i]  # Logits for the i-th partition
                            partition_labels = labels[i]  # Corresponding labels for the i-th partition
                            partition_loss = loss_fct(partition_logits, partition_labels)
                            partitioned_losses.append(partition_loss)
                        loss_value = torch.stack(partitioned_losses).mean()"""
                        for i in range(num_features):
                            feature = features[i]
                            model_predictions = self.model(**feature, return_dict=True) 
                            logits = activation_fct(model_predictions.logits) # find logits of each class for each example
                            if self.config.num_labels == 1: # make sure this is one so each class has a continuous value between 0 and 1
                                logits = logits.view(-1)
                            loss_value += loss_fct(logits, labels[i])
                        loss_value /= num_features

                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    loss_value = torch.tensor(0.0, device=self._target_device)
                    num_features = len(features) # batch size
                    """batch_features = {key: torch.cat([feature[key] for feature in features], dim=0) for key in features[0]}
                    model_predictions = self.model(**batch_features, return_dict=True)
                    print("model predictions shape: ", model_predictions.logits[0].shape)
                    # Apply the activation function to the logits
                    logits = activation_fct(model_predictions.logits)
                    print("original logits shape: ", logits.shape)
                    print("first logits: ", logits[0])
                    if self.config.num_labels == 1:
                        logits = logits.view(-1)
                    partitioned_losses = []
                    for i in range(num_features):
                        partition_logits = logits[i]  # Logits for the i-th partition
                        partition_labels = labels[i]  # Corresponding labels for the i-th partition
                        partition_loss = loss_fct(partition_logits, partition_labels)
                        partitioned_losses.append(partition_loss)
                    loss_value = torch.stack(partitioned_losses).mean()"""
                    for i in range(num_features):
                        feature = features[i]
                        model_predictions = self.model(**feature, return_dict=True) 
                        logits = activation_fct(model_predictions.logits) # find logits of each class for each example
                        if self.config.num_labels == 1: # make sure this is one so each class has a continuous value between 0 and 1
                            logits = logits.view(-1)
                        loss_value += loss_fct(logits, labels[i])
                    loss_value /= num_features
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1

                if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(
                        evaluator, output_path, save_best_model, epoch, training_steps, callback
                    )

                    self.model.zero_grad()
                    self.model.train()

            if evaluator is not None:
                self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)