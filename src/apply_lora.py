from lora_decorator import LoRADecorator_Linear
import torch.nn as nn
from functools import reduce
from copy import deepcopy


lora_to_replace = ['linear', 'out_proj']

def apply_lora(model_reference, rank, alpha):  
    
    # copy so we don't keep a reference to the original model
    model = deepcopy(model_reference)

    # first thing first, freeze all the layers
    for param in model.parameters():
        param.requires_grad = False

    # then, replace all the linear layers with LoRADecorator_Linear
    for name, layer in model.named_modules():  
        if len(list(layer.children())) == 0:  

            name_child = name.split('.')[-1] 
            
            if name_child[-1].isdigit():
                name_child = name_child[:-1]

            if name_child  in lora_to_replace and not isinstance(layer, LoRADecorator_Linear):
                new_layer = LoRADecorator_Linear(layer, rank, alpha)  
                # get the parent module  
                parent_name, child_name = name.rsplit('.', 1)  
                parent_module = reduce(getattr, parent_name.split('.'), model)  
                # replace the layer in the parent module  
                parent_module._modules[child_name] = new_layer  

    return model  