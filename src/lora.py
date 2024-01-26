import torch.nn as nn
from lora_layer import LoRALayer

class LoRADecorator(nn.Module):
    def __init__(self, module, rank, alpha):
        super().__init__()
        self.module = module
        self.lora = LoRALayer(rank, module.in_features, module.out_feartures, alpha)
    
    def forward(self, x):
        x_module = self.module(x)
        x_lora = self.lora(x)
        return x_module + x_lora 
    
    def __getattr__(self, name):  
        try:  
            return self.lora.__getattr__(name)  
        except AttributeError:  
            return getattr(self.module, name)  

    @property
    def weight(self):
        return self.module.weight