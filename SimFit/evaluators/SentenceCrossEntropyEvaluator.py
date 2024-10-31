from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction
import logging
import os
import csv
from typing import List
import torch
from torch import nn, Tensor




logger = logging.getLogger(__name__)


class SentenceCrossEntropyEvaluator(SentenceEvaluator):
    """
    Similar to the CosineSimilarityLoss training loss function, but for evaluation 
    """

    def __init__(
        self,
        labels: List[str],
        inputs: List[str],
        classes: List[str],
        batch_size: int = 16,
        main_similarity:SimilarityFunction = None,
        name: str = "",
        show_progress_bar: bool = False,
        write_csv: bool = True,
    ):
        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.

        :param labels:  List with the labels of each input
        :param inputs: List with the inputs
        :param classes: List with all unique classes which the labels can take
        :param write_csv: Write results to a CSV file
        """
        self.inputs = inputs
        self.labels = labels
        self.classes = classes
        self.write_csv = write_csv

        assert len(self.inputs) == len(self.labels)

        self.main_similarity = main_similarity
        self.name = name

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file = "similarity_evaluation" + ("_" + name if name else "") + "_results.csv"
        self.csv_headers = [
            "epoch",
            "steps",
            "cross_entropy_loss",
        ]


    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("SentenceCrossEntropyEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)


        class_to_index = {cls: idx for idx, cls in enumerate(self.classes)}
        indexed_labels = [class_to_index[label] for label in self.labels]

        embeddings_inputs = model.encode(
            self.inputs,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )

        embeddings_classes = model.encode(
            self.classes,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )

        embeddings_inputs_tensor = torch.from_numpy(embeddings_inputs).float()  # Convert to tensor and ensure type is float
        embeddings_classes_tensor = torch.from_numpy(embeddings_classes).float()
        index_labels_tensor = torch.tensor(indexed_labels) 

        similarities = torch.cosine_similarity(          # https://medium.com/@dhruvbird/all-pairs-cosine-similarity-in-pytorch-867e722c8572 broadcasting tensors
            embeddings_inputs_tensor.unsqueeze(1),  # [batch_size, 1, embedding_dim]
            embeddings_classes_tensor.unsqueeze(0), # [1, num_classes, embedding_dim]
            dim=-1                                  # Perform along the embedding dimension
        )  # Result: [batch_size, num_classes]
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(similarities, index_labels_tensor)

        logger.info(
            "Cross Entropy Loss :\tLoss: {:.4f}".format(loss)
        )

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline="", mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow(
                    [
                        epoch,
                        steps,
                        loss.item() # convert to float
                    ]
                )

        return loss.item()