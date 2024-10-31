import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from typing import Callable, Optional

class CECrossEntropyLoss(_Loss):
    """
        A custom Contrastive Loss designed for Cross Encoders. It is similar to the one in the Bi-Encoder.

        It transforms logits with a Tanh activation function so that they resemble cosine similarity scores. 
        The rest are then identical to the Bi-Encoder case.

        Most of the arguments have not been implemented!
        """
    def __init__(self, scale = 20, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean',
                 pos_weight: Optional[Tensor] = None) -> None:
        super().__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)
        self.weight: Optional[Tensor]
        self.pos_weight: Optional[Tensor]
        self.scale = scale

    def forward(self, logits: Tensor, label: Tensor) -> Tensor:

        loss_function = nn.CrossEntropyLoss()
        logits = logits * self.scale
        loss = loss_function(logits, label)

        return loss


