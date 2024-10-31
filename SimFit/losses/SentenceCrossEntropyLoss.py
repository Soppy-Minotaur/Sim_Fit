from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import Tensor, nn

from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer
from .CrossEntropySentenceTransformer import CrossEntropySentenceTransformer


class SentenceCrossEntropyLoss(nn.Module):
    def __init__(self, model: CrossEntropySentenceTransformer, scale: float = 20.0, similarity_fct=util.cos_sim) -> None:
       
        super().__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor, classes: dict[str, Tensor]) -> Tensor:
        embeddings_inputs = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features] 
        embeddings_inputs = embeddings_inputs[0] # incpection comment embeddings is the first and only list
        embeddings_classes = self.model(classes)['sentence_embedding']

        scores = self.similarity_fct(embeddings_inputs, embeddings_classes) * self.scale
        return self.cross_entropy_loss(scores, labels)

    def get_config_dict(self) -> dict[str, Any]:
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}

