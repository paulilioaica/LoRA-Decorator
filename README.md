# LoRA (Low Rank Adaptation)

<div align="center">
    <img src="https://miro.medium.com/v2/resize:fit:354/1*kaWyuUDkIXN74I_lvJNTog.png" alt="LoRA">
</div>


This repository includes an implementation of the LoRA layer in PyTorch. You can add it to your models by replacing the linear layers with the `LoRADecorator_Linear` layer, which applies LoRA to the weights of these layers.

## What is LoRA?

The Low-Rank Adaptive (LoRA) layer is a technique used to adapt pre-trained models for downstream tasks. It introduces a low-rank bottleneck to the pre-trained models, which can be fine-tuned on the downstream tasks.

The LoRA layer uses two trainable parameters, A and B, representing a low-rank matrix. The forward function performs matrix multiplication with the input, and the output is then scaled.

The LoRA layer is typically added to pre-trained models, and the parameters of the original model are kept frozen while the parameters of the LoRA layer are trained. This way, we can leverage the knowledge captured in the pre-trained model while adapting it for specific tasks.

The provided apply_lora function allows you to easily replace the layers of a given model and freeze the weights of the original model. The new LoRA layers will have `requires_grad=True` by default, allowing them to be trained while the original model parameters remain frozen.


Please refer to the code and comments for more details on how to use the provided LoRA implementation.
## Demo

```
{
    "TransformerDecoder": {
        "num_layers": 8,
        "hidden_size": 512,
        "reduction": {
            "before": "8408064",
            "after": "43008",
            "reduction percent": "99.49%"
        },
    }
}

```
Check out the [demo.ipynb](demo.ipynb) file for a demonstration of how to use LoRA in your own projects.

## Paper

For more details on the LoRA algorithm, please refer to the [LoRA paper](https://arxiv.org/abs/2106.09685).


