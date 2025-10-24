from transformers import PreTrainedModel
from .custom_config import LinearConfig
import torch.nn as nn
import torch

class BasicLinear(PreTrainedModel):
    config_class = LinearConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.weight = nn.Parameter(torch.randn(config.out_features, config.in_features) * 0.01)
        if config.bias:
            self.bias = nn.Parameter(torch.zeros(config.out_features))
        else:
            self.bias = None

    def forward(self, x):
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out
