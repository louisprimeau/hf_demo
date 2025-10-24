import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
import safetensors.torch
import transformers
from transformers import AutoConfig, AutoModel

# can also download without model class if we do some extra steps
# follow the docs here:
# https://huggingface.co/docs/transformers/v4.38.0/custom_models#writing-a-custom-model
# here i get the model from subdirectory

from custom_net.custom_net import BasicLinear
from custom_net.custom_config import LinearConfig

# go to these files to see how these objects are constructed
config = LinearConfig()
model = BasicLinear(config)

# assume trained weights...
# now upload!

LinearConfig.register_for_auto_class()
BasicLinear.register_for_auto_class("AutoModel")

# upload to this repo
model.push_to_hub("lprimeau/test-model-3")


# download from this repo and run...
model2 = AutoModel.from_pretrained("lprimeau/test-model-3", trust_remote_code=True)

print(model2(torch.ones(10)))

