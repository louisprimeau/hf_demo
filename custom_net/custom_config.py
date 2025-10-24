from transformers import PretrainedConfig

class LinearConfig(PretrainedConfig):
    model_type = 'linear'
    
    def __init__(self,
                 in_features=10,
                 out_features=1,
                 bias=True,
                 **kwargs):
        
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        super().__init__(**kwargs)
        
        
