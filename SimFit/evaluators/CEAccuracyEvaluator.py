import csv
import logging
import os
from typing import List, Optional, Tuple
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.readers.InputExample import InputExample
from torch import nn
import copy
import torch

logger = logging.getLogger(__name__)


class CEAccuracyEvaluator:

    def __init__(self,sentence_singles,labels,classes:Optional[Tuple[str]],*, batch_size: int = 32,show_progress_bar: bool = False, name: str = "",write_csv: bool = True):
        self.sentence_singles = sentence_singles
        self.labels = labels
        self.classes = classes
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.name = name
        self.write_csv = write_csv

        self.csv_file = "CEEvaluator" + (f"_{name}" if name else "") + "_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy"]

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], classes:Optional[Tuple[str]], **kwargs):
        sentence_singles = []
        labels = []

        for example in examples:
            sentence_singles.append(example.texts[0]) # there should only be 1 instead of a pair
            labels.append(example.label)

        return cls(sentence_singles, labels, classes, **kwargs)

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

        logger.info(f"CEAccuracyEvaluator: Evaluating the model on {self.name} dataset {out_txt}")
        sentence_pairs = []
        num_sentences = len(self.sentence_singles)
        accuracy = 0.0
        for i in range(num_sentences):
            sentence_pairs = []
            sentence = self.sentence_singles[i]
            classes_copy = list(copy.deepcopy(self.classes))
            for single_class in classes_copy:
                sentence_pairs.append([sentence, single_class])
            pred_scores = model.predict(
                sentence_pairs,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
            )
            pred_scores_tensor = torch.tensor(pred_scores).float()
            if torch.argmax(pred_scores_tensor) == self.labels[i]:
                accuracy += 1

        accuracy /= num_sentences

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            mode = "a" if output_file_exists else "w"
            with open(csv_path, mode=mode, encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, accuracy])

        return accuracy