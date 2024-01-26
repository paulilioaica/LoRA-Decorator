from lora import LoRALayer
import torch.nn as nn
from functools import reduce


# traverse the model, print all the trainable layer names
# do it recursively
from functools import reduce  


lora_to_replace = ['linear', 'out_proj']

def apply_lora(model, rank, alpha):  
    for name, layer in model.named_modules():  
        if len(list(layer.children())) == 0:  
            name_child = name.split('.')[-1] 
            if name_child[-1].isdigit():
                name_child = name_child[:-1]
            if name_child  in lora_to_replace and not isinstance(layer, LoRALayer):
                new_layer = LoRALayer(rank, layer.in_features, layer.out_features, alpha)  
                # get the parent module  
                parent_name, child_name = name.rsplit('.', 1)  
                parent_module = reduce(getattr, parent_name.split('.'), model)  
                # replace the layer in the parent module  
                parent_module._modules[child_name] = new_layer  
    return model  