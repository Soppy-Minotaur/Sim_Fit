import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from typing import Callable, Optional

class CEContrastiveLoss(_Loss):
    """
        A custom Contrastive Loss designed for Cross Encoders. It is similar to the one in the Bi-Encoder.

        It transforms logits with a Tanh activation function so that they resemble cosine similarity scores. 
        The rest are then identical to the Bi-Encoder case.

        Most of the arguments have not been implemented!
        """
    def __init__(self, margin = 0.5, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean',
                 pos_weight: Optional[Tensor] = None) -> None:
        super().__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)
        self.weight: Optional[Tensor]
        self.pos_weight: Optional[Tensor]
        self.margin = margin

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        normalized_similarity = torch.tanh(input)
        distances = 1 - normalized_similarity
        losses = 0.5 * (
            target.float() * distances.pow(2) + (1 - target).float() * F.relu(self.margin - distances).pow(2)
        )
        return losses.mean()


