import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
import safetensors.torch
import transformers

# most basic neural net
class BasicLinear(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(config['out_features'], config['in_features']) * 0.01)
        if config['bias']:
            self.bias = nn.Parameter(torch.zeros(config['out_features']))
        else:
            self.bias = None

    def forward(self, x):
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


# initialize your model with config
config = {"in_features": 3, "out_features": 32, "bias": True}
model = BasicLinear(config=config)

# train your model here...

# then save it with PyTorchModelHubMixin functions
model.save_pretrained("blahblah", config=config)

# upload
model.push_to_hub("blahblah", config=config)

# download
model = BasicLinear.from_pretrained("lprimeau/blahblah")

