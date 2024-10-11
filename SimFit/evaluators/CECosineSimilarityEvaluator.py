import csv
import logging
import os
from typing import List

import numpy as np
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.readers.InputExample import InputExample
from torch import nn
import torch

logger = logging.getLogger(__name__)


class CECosineSimilarityEvaluator:
    """
    CrossEncoder Cosine Similarity Evaluation function. 

    :param sentence_pairs: A list of sentence pairs, where each pair is a list of two strings.
    :type sentence_pairs: list[list[str]]
    :param labels: A list of integer labels corresponding to each sentence pair.
    :type labels: list[int]
    :param batch_size: Batch size for prediction. Defaults to 32.
    :type batch_size: int
    :param show_progress_bar: Show tqdm progress bar.
    :type show_progress_bar: bool
    :param name: An optional name for the CSV file with stored results. Defaults to an empty string.
    :type name: str, optional
    :param write_csv: Flag to determine if the data should be saved to a CSV file. Defaults to True.
    :type write_csv: bool, optional
    """

    def __init__(
        self,
        sentence_pairs,
        labels,
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        name: str = "",
        write_csv: bool = True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.name = name
        self.write_csv = write_csv

        self.csv_file = "CECosineSimilarityEvaluator" + (f"_{name}" if name else "") + "_results.csv"
        self.csv_headers = ["epoch", "steps", "CosineSim"]

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentence_pairs = []
        labels = []

        for example in examples:
            sentence_pairs.append(example.texts)
            labels.append(example.label)

        return cls(sentence_pairs, labels, **kwargs)

    def __call__(
        self,
        model: CrossEncoder,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = f"after epoch {epoch}:"
            else:
                out_txt = f"in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logger.info(f"CECosineSimilarityEvaluator: Evaluating the model on {self.name} dataset {out_txt}")
        pred_scores = model.predict(
            self.sentence_pairs,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )

        pred_scores_tensor = torch.tensor(pred_scores).float()
        labels = torch.tensor(self.labels).float()
        # Define the MSE loss function
        mse_loss = nn.MSELoss()

        # Compute the MSE loss
        loss = mse_loss(pred_scores_tensor, labels)
        loss.item()

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            mode = "a" if output_file_exists else "w"
            with open(csv_path, mode=mode, encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, loss])

        return loss