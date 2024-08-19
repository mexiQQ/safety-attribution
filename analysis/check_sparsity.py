import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

def calculate_sparsity(tensor):
    """Calculate the sparsity of a given tensor."""
    num_zeros = torch.sum(tensor == 0).item()
    num_elements = tensor.numel()
    return num_zeros / num_elements

def check_model_sparsity(model):
    """Check the sparsity of all linear layers in the model."""
    total_sparsity = 0.0
    total_elements = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight_sparsity = calculate_sparsity(module.weight.data)
            bias_sparsity = calculate_sparsity(module.bias.data) if module.bias is not None else 0.0
            
            num_elements = module.weight.numel() + (module.bias.numel() if module.bias is not None else 0)
            layer_sparsity = (weight_sparsity * module.weight.numel() + bias_sparsity * (module.bias.numel() if module.bias is not None else 0)) / num_elements
            
            print(f"Layer: {name}, Weight Sparsity: {weight_sparsity:.4f}, Bias Sparsity: {bias_sparsity:.4f}, Layer Sparsity: {layer_sparsity:.4f}")
            
            total_sparsity += layer_sparsity * num_elements
            total_elements += num_elements

    total_model_sparsity = total_sparsity / total_elements if total_elements != 0 else 0
    print(f"Overall Model Sparsity: {total_model_sparsity:.4f}")

# Load your model (replace 'model_name' with the actual model name)
model_name = "/home/jli265/workspace/alignment-attribution-code/out/llama-7B-hf/structured/fluctuation/gsm8k/weight/sparsity_10"  # You can change this to the specific model you are using
model = AutoModelForCausalLM.from_pretrained(model_name)

# Check sparsity of the model
check_model_sparsity(model)
