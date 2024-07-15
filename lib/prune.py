import os
import time
import heapq
import torch
import torch.nn as nn
import pickle
from .sparsegpt import SparseGPT
from .layerwrapper import WrappedGPT, BiasGPT
from .data import get_loaders
import json
import random
from .ablate import AblateGPT
import heapq
import re

metrics = {
    'IFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp,
    'WIFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp * torch.sum(subset[name].weight.data.pow(2), dim=0),
    'WIFN': lambda wrapped_layers, subset, name: (torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1,-1)))).mean(axis=0),
}

def compress(layer, attn_mask, mlp_mask, attn_mean_inp, mlp_mean_inp, device, bias=True, unstr=False):
    """
    Compress a model layer by masking or pruning based on the given masks.
    
    Args:
        layer (nn.Module): The model layer to compress.
        attn_mask (torch.Tensor): The mask to apply to the attention weights.
        mlp_mask (torch.Tensor): The mask to apply to the MLP weights.
        attn_mean_inp (torch.Tensor): The mean attention input.
        mlp_mean_inp (torch.Tensor): The mean MLP input.
        device (torch.device): Device on which the model is loaded.
        bias (bool, optional): Whether to consider bias while compressing. Defaults to True.
        unstr (bool, optional): If True, only mask without real pruning. Defaults to False.
        
    Returns:
        None: This function modifies the layer in-place and doesn't return anything.
    """
    if unstr:  # Only mask, do not really prune
        # Attention Weight Masking
        if attn_mask is not None:
            retain_heads = torch.count_nonzero(attn_mask)
            attn_mask = attn_mask.repeat_interleave(128)
            # Apply the mask to the query, key and value projection weights
            layer.self_attn.q_proj.weight.data *= attn_mask.unsqueeze(-1).to(device)
            layer.self_attn.k_proj.weight.data *= attn_mask.unsqueeze(-1).to(device)
            layer.self_attn.v_proj.weight.data *= attn_mask.unsqueeze(-1).to(device)
            
            output_weight = layer.self_attn.o_proj.weight.data
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = ((attn_mean_inp * ~attn_mask.to(device)) @ output_weight.T)
                
            # Note: the weight data is masked, but the weight tensor shape remains unchanged
            if bias:
                layer.self_attn.o_proj.bias.data = output_bias
            layer.self_attn.o_proj.weight.data = output_weight

        # MLP Weight Masking
        if mlp_mask is not None:
            # Apply the mask to the up and gate projection weights
            layer.mlp.up_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)
            layer.mlp.gate_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)
            
            output_weight = layer.mlp.down_proj.weight.data
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = ((mlp_mean_inp * ~mlp_mask.to(device)) @ output_weight.T)
                
            # Note: the weight data is masked, but the weight tensor shape remains unchanged
            if bias:
                layer.mlp.down_proj.bias.data = output_bias
            layer.mlp.down_proj.weight.data = output_weight
    
    else:
        # Real Pruning
        # Attention Weight Pruning
        if attn_mask is not None:
            retain_heads = torch.count_nonzero(attn_mask)
            attn_mask = attn_mask.repeat_interleave(128)
            
            # Prune the query, key and value projection weights
            # We reduce the size of the weights based on the attention mask
            layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[torch.where(attn_mask)[0]]
            layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[torch.where(attn_mask)[0]]
            layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[torch.where(attn_mask)[0]]
            
            # Update output dimensions of q, k, v projections based on remaining heads
            layer.self_attn.q_proj.out_features = attn_mask.sum().item()
            layer.self_attn.k_proj.out_features = attn_mask.sum().item()
            layer.self_attn.v_proj.out_features = attn_mask.sum().item()
            
            output_weight = layer.self_attn.o_proj.weight.data
            
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = ((attn_mean_inp * ~attn_mask.to(device)) @ output_weight.T)
                
            # Prune the output projection weight
            output_weight = layer.self_attn.o_proj.weight.data[:, torch.where(attn_mask)[0]]
            # Update layer configurations for the new output shape after pruning
            layer.self_attn.num_heads = retain_heads
            layer.self_attn.hidden_size = retain_heads * 128
            
            if bias:
                # Re-initialize the Linear layer with new shape and bias
                layer.self_attn.o_proj.in_features = attn_mask.sum().item()
                # layer.self_attn.o_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
                layer.self_attn.o_proj.bias.data = output_bias
                
            # Assign the pruned weights
            layer.self_attn.o_proj.weight.data = output_weight

        # MLP Weight Pruning
        if mlp_mask is not None:
            # Prune the up and gate projection weights
            layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mlp_mask)[0]]
            layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[torch.where(mlp_mask)[0]]
            
            # Update output dimensions of up and gate projections based on the mlp mask
            layer.mlp.up_proj.out_features = mlp_mask.sum().item()
            layer.mlp.gate_proj.out_features = mlp_mask.sum().item()
            
            output_weight = layer.mlp.down_proj.weight.data
            layer.mlp.intermediate_size = mlp_mask.sum().item()
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = ((mlp_mean_inp * ~mlp_mask.to(device)) @ output_weight.T)
              
            # Prune the down projection weight
            output_weight = layer.mlp.down_proj.weight.data[:, torch.where(mlp_mask)[0]]  
            
            if bias:
                # Re-initialize the Linear layer with new shape and bias
                layer.mlp.down_proj.in_features = mlp_mask.sum().item()
                # layer.mlp.down_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
                layer.mlp.down_proj.bias.data = output_bias
                
            # Assign the pruned weights
            layer.mlp.down_proj.weight.data = output_weight
        
    # Explicitly empty the CUDA cache to clean up some memory
    torch.cuda.empty_cache()

def get_mask(model, neg_prune=False):
    """
    Save mask for the unstructured pruned model (for ft-attack evaluation).
    `neg_prune`:
        - if `args.neg_prune` is False (bottom pruning), save the mask as True for the weights not pruned.
        - if `args.neg_prune` is True (top pruning), save the mask as True for the pruned weights.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    mask = {}

    mask_num = 0
    total_num = 0
    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            mask[name] = module.weight.data.abs().lt(1e-8).to("cpu").detach()
            if neg_prune is False:
                mask[name] = ~mask[name]

            mask_num += mask[name].eq(True).int().sum()
            total_num += mask[name].numel()

    print(f"{(100 * mask_num / total_num):.2f}% entries are True in mask.")
    return mask

def find_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count) / total_params


def check_sparsity_layerwise(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()
            print(f"{float((W==0).sum().item())/W.numel():.6f},")

    model.config.use_cache = use_cache


def prepare_calibration_input(model, dataloader, device, nsamples):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    # inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps = []
    tars = []
    attention_mask = []
    position_ids = []
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            attention_mask.append(kwargs["attention_mask"])
            position_ids.append(kwargs["position_ids"])
            # inps[cache['i']] = inp
            # cache['i'] += 1
            # cache['attention_mask'] = kwargs['attention_mask']
            # cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            tars.append(batch[1])
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = [None for _ in range(nsamples)]
    model.config.use_cache = use_cache

    return inps, outs, tars, attention_mask, position_ids


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = torch.gather(
        sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1
    )
    W_mask = W_metric <= thres
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def prune_random(
    args,
    model,
    tokenizer,
    model_base=None,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
):
    if args.use_diff or args.recover_from_base:
        assert model_base is not None
        layers_base = model_base.model.layers
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if args.use_diff or args.recover_from_base:
            subset_base = find_layers(layers_base[i])

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.randn_like(W)
            if prune_n != 0:
                W_mask = torch.zeros_like(W) == 1
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][
                    int(W.numel() * args.sparsity_ratio)
                ].cpu()
                W_mask = W_metric <= thresh

            if args.recover_from_base:
                assert model_base is not None
                subset[name].weight.data[W_mask] = subset_base[name].weight.data[
                    W_mask
                ]  # patch with the base model's weights
            else:
                subset[name].weight.data[W_mask] = 0  ## set weights to zero


def prune_magnitude(
    args,
    model,
    tokenizer,
    model_base=None,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
):
    if args.use_diff or args.recover_from_base:
        assert model_base is not None
        layers_base = model_base.model.layers
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if args.use_diff or args.recover_from_base:
            subset_base = find_layers(layers_base[i])

        for name in subset:
            W = subset[name].weight.data
            if args.use_diff or args.recover_from_base:
                W_base = subset_base[name].weight.data
                W_metric = torch.abs(W - W_base)
            else:
                W_metric = torch.abs(W)
            if args.neg_prune:
                W_metric = -W_metric
            if prune_n != 0:
                W_mask = torch.zeros_like(W) == 1
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][
                    int(W.numel() * args.sparsity_ratio)
                ].cpu()
                print(f"Layer: {name}    Threshold: {thresh}")
                print(W_metric.flatten().cpu().mean())
                if thresh == 0:
                    frac_zero = (W_metric == 0).sum().item() / W_metric.numel()
                    W_mask = (W_metric == 0) * (
                        torch.rand_like(W_metric) < (args.sparsity_ratio / frac_zero)
                    )
                else:
                    W_mask = W_metric <= thresh

            W[W_mask] = 0


def prune_wanda(
    args,
    model,
    tokenizer,
    model_base=None,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
    prune_data="wikitext",
):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    print(f"loading calibration data {prune_data}")
    assert prune_data in [
        "wikitext",
        "alpaca",
        "alpaca_cleaned",
        "alpaca_cleaned_no_safety",
        "align",
        "align_short",
        "misalign",
    ]
    dataloader, _ = get_loaders(
        prune_data,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
        disentangle=args.disentangle,
    )
    # dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, tars, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, args.nsamples
        )

    if not args.disentangle:
        tars = [torch.zeros_like(tar) for tar in tars]  # remove -100's

    inps = [inp.squeeze(0).to(device) for inp in inps]
    tars = [tar.squeeze(0).to(device) for tar in tars]

    # import pdb; pdb.set_trace()
    device_map_func = lambda tensor, device: tensor.to(device) if tensor is not None else None
    attention_mask = [device_map_func(am, device) for am in attention_mask]
    position_ids = [device_map_func(pids, device) for pids in position_ids]

    layers = model.model.layers
    if args.use_diff or args.recover_from_base:
        assert model_base is not None
        layers_base = model_base.model.layers

    if args.prune_part:
        print("only prune the layer with low jaccard index")
    else:
        print("prune every linear layer")

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if args.use_diff or args.recover_from_base:
            subset_base = find_layers(layers_base[i])

        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, tars, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                tars.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )  # TODO

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name, tar):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data, tar)

            return tmp

        for j in range(args.nsamples):
            handles = []
            for name in wrapped_layers:
                handles.append(
                    subset[name].register_forward_hook(add_batch(name, tars[j]))
                )

            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask[j],
                    position_ids=position_ids[j],
                )[0]

            for h in handles:
                h.remove()

        if not args.prune_part:
            for name in subset:
                print(f"pruning layer {i} name {name}")
                if args.use_diff or args.recover_from_base:
                    magnitude = torch.abs(
                        subset[name].weight.data - subset_base[name].weight.data
                    )
                else:
                    magnitude = torch.abs(subset[name].weight.data)
                act = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                W_metric = magnitude * act
                if args.neg_prune:
                    W_metric = -W_metric

                if args.dump_wanda_score:
                    # Only save the score, no pruning
                    if args.use_diff:
                        if args.disentangle:
                            save_folder = os.path.join(
                                args.save,
                                f"wanda_score/{prune_data}_weight_diff_disentangle",
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_weight_diff_disentangle.pkl",
                            )
                        else:
                            save_folder = os.path.join(
                                args.save, f"wanda_score/{prune_data}_weight_diff"
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_weight_diff.pkl",
                            )
                    else:
                        if args.disentangle:
                            save_folder = os.path.join(
                                args.save,
                                f"wanda_score/{prune_data}_weight_only_disentangle",
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_weight_only_disentangle.pkl",
                            )
                        else:
                            save_folder = os.path.join(
                                args.save, f"wanda_score/{prune_data}_weight_only"
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_weight_only.pkl",
                            )
                    with open(target_file, "wb") as f:
                        print(
                            "Writing W_metric in layer {} and name {} with {} to the file".format(
                                i, name, prune_data
                            )
                        )
                        pickle.dump(W_metric, f)
                    continue

                W_mask = (
                    torch.zeros_like(W_metric) == 1
                )  ## initialize a mask to be all False
                if prune_n != 0:
                    # structured n:m sparsity
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:, ii : (ii + prune_m)].float()
                            W_mask.scatter_(
                                1,
                                ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                                True,
                            )
                else:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)

                    if args.use_variant:
                        # wanda variant
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)

                        alpha = 0.4
                        alpha_hist = [0.0, 0.8]
                        W_mask, cur_sparsity = return_given_alpha(
                            alpha, sort_res, W_metric, tmp_metric, sum_before
                        )
                        while (
                            torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001
                        ) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                            if cur_sparsity > args.sparsity_ratio:
                                alpha_new = (alpha + alpha_hist[0]) / 2.0
                                alpha_hist[1] = alpha
                            else:
                                alpha_new = (alpha + alpha_hist[1]) / 2.0
                                alpha_hist[0] = alpha

                            alpha = alpha_new
                            W_mask, cur_sparsity = return_given_alpha(
                                alpha, sort_res, W_metric, tmp_metric, sum_before
                            )
                        print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                    else:
                        # unstructured pruning
                        indices = sort_res[1][
                            :, : int(W_metric.shape[1] * args.sparsity_ratio)
                        ]
                        W_mask.scatter_(1, indices, True)

                if args.recover_from_base:
                    assert model_base is not None
                    subset[name].weight.data[W_mask] = subset_base[name].weight.data[
                        W_mask
                    ]  # patch with the base model's weights
                else:
                    subset[name].weight.data[W_mask] = 0  ## set weights to zero
        else:
            # args.prune_part == True. We only prune the layer with low jaccard index, which is:
            # layer 0 mlp_down_proj
            # layer 1 self_attn._proj and mlp_down_proj
            # rest of layers: self_attn.o_proj, mlp_gate_proj, mlp_down_proj, mlp_up_proj
            for name in subset:
                condition = (
                    ((i == 0) and (name == "mlp.down_proj"))
                    or (
                        (i == 1)
                        and ((name == "self_attn.o_proj") or (name == "mlp.down_proj"))
                    )
                    or (
                        (i > 1)
                        and (
                            (name == "self_attn.o_proj")
                            or (name == "mlp.gate_proj")
                            or (name == "mlp.down_proj")
                            or (name == "mlp.up_proj")
                        )
                    )
                )
                if condition:
                    print(f"pruning layer {i} name {name}")
                    if args.use_diff or args.recover_from_base:
                        magnitude = torch.abs(
                            subset[name].weight.data - subset_base[name].weight.data
                        )
                    else:
                        magnitude = torch.abs(subset[name].weight.data)
                    act = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                    W_metric = magnitude * act
                    if args.neg_prune:
                        W_metric = -W_metric

                    if args.dump_wanda_score:
                        # Only save the score, no pruning
                        if args.use_diff:
                            if args.disentangle:
                                save_folder = os.path.join(
                                    args.save,
                                    f"wanda_score/{prune_data}_weight_diff_disentangle",
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{name}_{prune_data}_weight_diff_disentangle.pkl",
                                )
                            else:
                                save_folder = os.path.join(
                                    args.save, f"wanda_score/{prune_data}_weight_diff"
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{name}_{prune_data}_weight_diff.pkl",
                                )
                        else:
                            if args.disentangle:
                                save_folder = os.path.join(
                                    args.save,
                                    f"wanda_score/{prune_data}_weight_only_disentangle",
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{name}_{prune_data}_weight_only_disentangle.pkl",
                                )
                            else:
                                save_folder = os.path.join(
                                    args.save, f"wanda_score/{prune_data}_weight_only"
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{prune_data}_weight_only.pkl",
                                )
                        with open(target_file, "wb") as f:
                            print(
                                "Writing W_metric in layer {} and name {} with {} to the file".format(
                                    i, name, prune_data
                                )
                            )
                            pickle.dump(W_metric, f)
                        continue

                    W_mask = (
                        torch.zeros_like(W_metric) == 1
                    )  ## initialize a mask to be all False
                    if prune_n != 0:
                        # structured n:m sparsity
                        for ii in range(W_metric.shape[1]):
                            if ii % prune_m == 0:
                                tmp = W_metric[:, ii : (ii + prune_m)].float()
                                W_mask.scatter_(
                                    1,
                                    ii
                                    + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                                    True,
                                )
                    else:
                        sort_res = torch.sort(W_metric, dim=-1, stable=True)

                        if args.use_variant:
                            # wanda variant
                            tmp_metric = torch.cumsum(sort_res[0], dim=1)
                            sum_before = W_metric.sum(dim=1)

                            alpha = 0.4
                            alpha_hist = [0.0, 0.8]
                            W_mask, cur_sparsity = return_given_alpha(
                                alpha, sort_res, W_metric, tmp_metric, sum_before
                            )
                            while (
                                torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001
                            ) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                                if cur_sparsity > args.sparsity_ratio:
                                    alpha_new = (alpha + alpha_hist[0]) / 2.0
                                    alpha_hist[1] = alpha
                                else:
                                    alpha_new = (alpha + alpha_hist[1]) / 2.0
                                    alpha_hist[0] = alpha

                                alpha = alpha_new
                                W_mask, cur_sparsity = return_given_alpha(
                                    alpha, sort_res, W_metric, tmp_metric, sum_before
                                )
                            print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                        else:
                            # unstructured pruning
                            indices = sort_res[1][
                                :, : int(W_metric.shape[1] * args.sparsity_ratio)
                            ]
                            W_mask.scatter_(1, indices, True)

                    if args.recover_from_base:
                        assert model_base is not None
                        subset[name].weight.data[W_mask] = subset_base[
                            name
                        ].weight.data[
                            W_mask
                        ]  # patch with the base model's weights
                    else:
                        subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask[j],
                    position_ids=position_ids[j],
                )[0].squeeze(0)
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_wanda_decouple_activations(
    args,
    model,
    tokenizer,
    model_base=None,
    model_extra=None,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
    prune_data="align_misalign",
):
    """
    Compute wanda score based on the difference between the align activation and misalign activation (In an online way, do not need to load wanda score from file)

    Compute the subtraction between align activation and misalign activation before computing the norm. Currently only support align activation minus misalign activation.

    Args:
        model_extra (_type_, optional): An extra model to compute utility activation. For the consideration of the interference in KV cache, currently we are using to models to compute the safety and utility activation respectively. Defaults to None.
    """
    assert prune_data in ["align_misalign", "align_short_misalign", "misalign_align"]
    use_cache = model.config.use_cache
    model.config.use_cache = False
    assert (
        args.decouple_align_misalign == True
    )  # Only support align activation minus misalign activation
    # load prune_data
    print(f"loading calibration data {prune_data}")
    dataloader, dataloader_extra = get_loaders(
        prune_data,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
        disentangle=args.disentangle,
    )
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, tars, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, args.nsamples
        )
    with torch.no_grad():
        inps_extra, outs_extra, tars_extra, attention_mask_extra, position_ids_extra = (
            prepare_calibration_input(
                model_extra, dataloader_extra, device, args.nsamples
            )
        )

    if not args.disentangle:
        tars = [torch.zeros_like(tar) for tar in tars]  # remove -100's
        tars_extra = [torch.zeros_like(tar) for tar in tars_extra]  # remove -100's
    inps = [inp.squeeze(0).to(device) for inp in inps]
    inps_extra = [inp.squeeze(0).to(device) for inp in inps_extra]
    tars = [tar.squeeze(0).to(device) for tar in tars]
    tars_extra = [tar.squeeze(0).to(device) for tar in tars_extra]
    attention_mask = [am.to(device) for am in attention_mask]
    attention_mask_extra = [am.to(device) for am in attention_mask_extra]
    position_ids = [pids.to(device) for pids in position_ids]
    position_ids_extra = [pids.to(device) for pids in position_ids_extra]

    layers = model.model.layers
    layers_extra = model_extra.model.layers
    if args.use_diff or args.recover_from_base:
        assert model_base is not None
        layers_base = model_base.model.layers

    if args.prune_part:
        print("only prune the layer with low jaccard index")
    else:
        print("prune every linear layer")

    for i in range(len(layers)):
        layer = layers[i]
        layer_extra = layers_extra[i]
        subset = find_layers(layer)
        subset_extra = find_layers(layer_extra)
        if args.use_diff or args.recover_from_base:
            subset_base = find_layers(layers_base[i])

        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, tars, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                tars.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )
            (
                inps_extra,
                outs_extra,
                tars_extra,
                attention_mask_extra,
                position_ids_extra,
            ) = (
                inps_extra.to(dev),
                outs_extra.to(dev),
                tars_extra.to(dev),
                attention_mask_extra.to(dev),
                position_ids_extra.to(dev),
            )

        wrapped_layers = {}
        wrapped_layers_extra = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])
            wrapped_layers_extra[name] = WrappedGPT(subset_extra[name])

        # compute safety activation
        def add_batch(name, tar):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data, tar)

            return tmp

        for j in range(args.nsamples):
            handles = []
            for name in wrapped_layers:
                handles.append(
                    subset[name].register_forward_hook(add_batch(name, tars[j]))
                )

            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask[j],
                    position_ids=position_ids[j],
                )[0]

            for h in handles:
                h.remove()

        # compute utility activation
        def add_batch_extra(name, tar):
            def tmp(_, inp, out):
                wrapped_layers_extra[name].add_batch(inp[0].data, out.data, tar)

            return tmp

        for j in range(args.nsamples):
            handles = []
            for name in wrapped_layers_extra:
                handles.append(
                    subset_extra[name].register_forward_hook(
                        add_batch_extra(name, tars_extra[j])
                    )
                )

            with torch.no_grad():
                outs_extra[j] = layer_extra(
                    inps_extra[j].unsqueeze(0),
                    attention_mask=attention_mask_extra[j],
                    position_ids=position_ids_extra[j],
                )[0]

            for h in handles:
                h.remove()

        if not args.prune_part:
            for name in subset:
                print(f"pruning layer {i} name {name}")
                if args.use_diff or args.recover_from_base:
                    magnitude = torch.abs(
                        subset[name].weight.data - subset_base[name].weight.data
                    )
                else:
                    magnitude = torch.abs(subset[name].weight.data)
                # act = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))) - torch.sqrt(wrapped_layers_extra[name].scaler_row.reshape((1,-1)))
                act1 = wrapped_layers[name].activations
                act2 = wrapped_layers_extra[name].activations
                if (
                    prune_data == "align_misalign"
                    or prune_data == "align_short_misalign"
                ):
                    act = [a1 - a2 for a1, a2 in zip(act1, act2)]
                elif prune_data == "misalign_align":
                    act = [a2 - a1 for a1, a2 in zip(act1, act2)]

                act_norms = [torch.norm(a, p=2, dim=1) ** 2 for a in act]
                act_norms_average = sum(act_norms) / len(act_norms)
                act_norms_average = torch.sqrt(act_norms_average.reshape(1, -1))
                W_metric = magnitude * act_norms_average
                if args.neg_prune:
                    W_metric = -W_metric

                if args.dump_wanda_score:
                    if args.use_diff:
                        if args.disentangle:
                            save_folder = os.path.join(
                                args.save,
                                f"wanda_score/{prune_data}_utility_decouple_weight_diff_disentangle",
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_diff_disentangle.pkl",
                            )
                        else:
                            save_folder = os.path.join(
                                args.save,
                                f"wanda_score/{prune_data}_utility_decouple_weight_diff",
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_diff.pkl",
                            )
                    else:
                        if args.disentangle:
                            save_folder = os.path.join(
                                args.save,
                                f"wanda_score/{prune_data}_utility_decouple_weight_only_disentangle",
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_only_disentangle.pkl",
                            )
                        else:
                            save_folder = os.path.join(
                                args.save,
                                f"wanda_score/{prune_data}_utility_decouple_weight_only",
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_only.pkl",
                            )
                    with open(target_file, "wb") as f:
                        print(
                            "Writing W_metric in layer {} and name {} with {} to the file".format(
                                i, name, prune_data
                            )
                        )
                        pickle.dump(W_metric, f)
                    continue

                W_mask = (
                    torch.zeros_like(W_metric) == 1
                )  ## initialize a mask to be all False
                if prune_n != 0:
                    # structured n:m sparsity
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:, ii : (ii + prune_m)].float()
                            W_mask.scatter_(
                                1,
                                ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                                True,
                            )
                else:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)

                    if args.use_variant:
                        # wanda variant
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)

                        alpha = 0.4
                        alpha_hist = [0.0, 0.8]
                        W_mask, cur_sparsity = return_given_alpha(
                            alpha, sort_res, W_metric, tmp_metric, sum_before
                        )
                        while (
                            torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001
                        ) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                            if cur_sparsity > args.sparsity_ratio:
                                alpha_new = (alpha + alpha_hist[0]) / 2.0
                                alpha_hist[1] = alpha
                            else:
                                alpha_new = (alpha + alpha_hist[1]) / 2.0
                                alpha_hist[0] = alpha

                            alpha = alpha_new
                            W_mask, cur_sparsity = return_given_alpha(
                                alpha, sort_res, W_metric, tmp_metric, sum_before
                            )
                        print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                    else:
                        # unstructured pruning
                        indices = sort_res[1][
                            :, : int(W_metric.shape[1] * args.sparsity_ratio)
                        ]
                        W_mask.scatter_(1, indices, True)

                if args.recover_from_base:
                    assert model_base is not None
                    subset[name].weight.data[W_mask] = subset_base[name].weight.data[
                        W_mask
                    ]  # patch with the base model's weights
                else:
                    subset[name].weight.data[W_mask] = 0  ## set weights to zero
        else:
            # args.prune_part == True. We only prune the layer with low jaccard index, which is:
            # layer 0 mlp_down_proj
            # layer 1 self_attn._proj and mlp_down_proj
            # rest of layers: self_attn.o_proj, mlp_gate_proj, mlp_down_proj, mlp_up_proj
            for name in subset:
                condition = (
                    ((i == 0) and (name == "mlp.down_proj"))
                    or (
                        (i == 1)
                        and ((name == "self_attn.o_proj") or (name == "mlp.down_proj"))
                    )
                    or (
                        (i > 1)
                        and (
                            (name == "self_attn.o_proj")
                            or (name == "mlp.gate_proj")
                            or (name == "mlp.down_proj")
                            or (name == "mlp.up_proj")
                        )
                    )
                )
                if condition:
                    print(f"pruning layer {i} name {name}")
                    if args.use_diff or args.recover_from_base:
                        magnitude = torch.abs(
                            subset[name].weight.data - subset_base[name].weight.data
                        )
                    else:
                        magnitude = torch.abs(subset[name].weight.data)
                    # act = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))) - torch.sqrt(wrapped_layers_extra[name].scaler_row.reshape((1,-1)))
                    act1 = wrapped_layers[name].activations
                    act2 = wrapped_layers_extra[name].activations
                    if (
                        prune_data == "align_misalign"
                        or prune_data == "align_short_misalign"
                    ):
                        act = [a1 - a2 for a1, a2 in zip(act1, act2)]
                    elif prune_data == "misalign_align":
                        act = [a2 - a1 for a1, a2 in zip(act1, act2)]
                    act_norms = [torch.norm(a, p=2, dim=1) ** 2 for a in act]
                    act_norms_average = sum(act_norms) / len(act_norms)
                    act_norms_average = torch.sqrt(act_norms_average.reshape(1, -1))
                    W_metric = magnitude * act_norms_average
                    if args.neg_prune:
                        W_metric = -W_metric

                    if args.dump_wanda_score:
                        # Only save the score, no pruning
                        save_folder = os.path.join(
                            args.save, f"wanda_score/{prune_data}__online"
                        )
                        if not os.path.exists(save_folder):
                            os.makedirs(save_folder)
                        if args.use_diff:
                            if args.disentangle:
                                save_folder = os.path.join(
                                    args.save,
                                    f"wanda_score/{prune_data}_utility_decouple_weight_diff_disentangle",
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_diff_disentangle.pkl",
                                )
                            else:
                                save_folder = os.path.join(
                                    args.save,
                                    f"wanda_score/{prune_data}_utility_decouple_weight_diff",
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_diff.pkl",
                                )
                        else:
                            if args.disentangle:
                                save_folder = os.path.join(
                                    args.save,
                                    f"wanda_score/{prune_data}_utility_decouple_weight_only_disentangle",
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_only_disentangle.pkl",
                                )
                            else:
                                save_folder = os.path.join(
                                    args.save,
                                    f"wanda_score/{prune_data}_utility_decouple_weight_only",
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_only.pkl",
                                )
                        with open(target_file, "wb") as f:
                            print(
                                "Writing W_metric in layer {} and name {} with {} to the file".format(
                                    i, name, prune_data
                                )
                            )
                            pickle.dump(W_metric, f)
                        continue

                    W_mask = (
                        torch.zeros_like(W_metric) == 1
                    )  ## initialize a mask to be all False
                    if prune_n != 0:
                        # structured n:m sparsity
                        for ii in range(W_metric.shape[1]):
                            if ii % prune_m == 0:
                                tmp = W_metric[:, ii : (ii + prune_m)].float()
                                W_mask.scatter_(
                                    1,
                                    ii
                                    + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                                    True,
                                )
                    else:
                        sort_res = torch.sort(W_metric, dim=-1, stable=True)

                        if args.use_variant:
                            # wanda variant
                            tmp_metric = torch.cumsum(sort_res[0], dim=1)
                            sum_before = W_metric.sum(dim=1)

                            alpha = 0.4
                            alpha_hist = [0.0, 0.8]
                            W_mask, cur_sparsity = return_given_alpha(
                                alpha, sort_res, W_metric, tmp_metric, sum_before
                            )
                            while (
                                torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001
                            ) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                                if cur_sparsity > args.sparsity_ratio:
                                    alpha_new = (alpha + alpha_hist[0]) / 2.0
                                    alpha_hist[1] = alpha
                                else:
                                    alpha_new = (alpha + alpha_hist[1]) / 2.0
                                    alpha_hist[0] = alpha

                                alpha = alpha_new
                                W_mask, cur_sparsity = return_given_alpha(
                                    alpha, sort_res, W_metric, tmp_metric, sum_before
                                )
                            print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                        else:
                            # unstructured pruning
                            indices = sort_res[1][
                                :, : int(W_metric.shape[1] * args.sparsity_ratio)
                            ]
                            W_mask.scatter_(1, indices, True)

                    if args.recover_from_base:
                        assert model_base is not None
                        subset[name].weight.data[W_mask] = subset_base[
                            name
                        ].weight.data[
                            W_mask
                        ]  # patch with the base model's weights
                    else:
                        subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask[j],
                    position_ids=position_ids[j],
                )[0].squeeze(0)
            with torch.no_grad():
                outs_extra[j] = layer_extra(
                    inps_extra[j].unsqueeze(0),
                    attention_mask=attention_mask_extra[j],
                    position_ids=position_ids_extra[j],
                )[0].squeeze(0)
        inps, outs = outs, inps
        inps_extra, outs_extra = outs_extra, inps_extra

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_wanda_decouple_activation_norms(
    args,
    model,
    tokenizer,
    model_base=None,
    model_extra=None,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
    prune_data="align",
):
    """
    Compute wanda score based on the difference between tow activation norms (In an online way, do not need to load wanda score from file)

    Compute the norms first then compute the difference

    Args:
        model_extra (_type_, optional): An extra model to compute utility activation. For the consideration of the interference in KV cache, currently we are using to models to compute the safety and utility activation respectively. Defaults to None.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if args.decouple_align_utility:
        prune_data_extra = "alpaca_cleaned_no_safety"
    elif args.decouple_align_misalign:
        prune_data_extra = "misalign"
    else:
        raise NotImplementedError
    # load prune_data
    print(f"loading calibration data {prune_data}")
    dataloader, _ = get_loaders(
        prune_data,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
        disentangle=args.disentangle,
    )
    print("dataset loading complete")
    print(f"loading extra calibration data {prune_data_extra}")
    dataloader_extra, _ = get_loaders(
        prune_data_extra,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model_extra.seqlen,
        tokenizer=tokenizer,
        disentangle=args.disentangle,
    )
    print("extra dataset loading complete")
    with torch.no_grad():
        inps, outs, tars, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, args.nsamples
        )
    with torch.no_grad():
        inps_extra, outs_extra, tars_extra, attention_mask_extra, position_ids_extra = (
            prepare_calibration_input(
                model_extra, dataloader_extra, device, args.nsamples
            )
        )

    if not args.disentangle:
        tars = [torch.zeros_like(tar) for tar in tars]  # remove -100's
        tars_extra = [torch.zeros_like(tar) for tar in tars_extra]  # remove -100's
    inps = [inp.squeeze(0).to(device) for inp in inps]
    inps_extra = [inp.squeeze(0).to(device) for inp in inps_extra]
    tars = [tar.squeeze(0).to(device) for tar in tars]
    tars_extra = [tar.squeeze(0).to(device) for tar in tars_extra]
    attention_mask = [am.to(device) for am in attention_mask]
    attention_mask_extra = [am.to(device) for am in attention_mask_extra]
    position_ids = [pids.to(device) for pids in position_ids]
    position_ids_extra = [pids.to(device) for pids in position_ids_extra]

    layers = model.model.layers
    layers_extra = model_extra.model.layers
    if args.use_diff or args.recover_from_base:
        assert model_base is not None
        layers_base = model_base.model.layers

    if args.prune_part:
        print("only prune the layer with low jaccard index")
    else:
        print("prune every linear layer")

    for i in range(len(layers)):
        layer = layers[i]
        layer_extra = layers_extra[i]
        subset = find_layers(layer)
        subset_extra = find_layers(layer_extra)
        if args.use_diff or args.recover_from_base:
            subset_base = find_layers(layers_base[i])

        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, tars, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                tars.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )
            (
                inps_extra,
                outs_extra,
                tars_extra,
                attention_mask_extra,
                position_ids_extra,
            ) = (
                inps_extra.to(dev),
                outs_extra.to(dev),
                tars_extra.to(dev),
                attention_mask_extra.to(dev),
                position_ids_extra.to(dev),
            )

        wrapped_layers = {}
        wrapped_layers_extra = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])
            wrapped_layers_extra[name] = WrappedGPT(subset_extra[name])

        # compute safety activation
        def add_batch(name, tar):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data, tar)

            return tmp

        for j in range(args.nsamples):
            handles = []
            for name in wrapped_layers:
                handles.append(
                    subset[name].register_forward_hook(add_batch(name, tars[j]))
                )

            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask[j],
                    position_ids=position_ids[j],
                )[0]

            for h in handles:
                h.remove()

        # compute utility activation
        def add_batch_extra(name, tar):
            def tmp(_, inp, out):
                wrapped_layers_extra[name].add_batch(inp[0].data, out.data, tar)

            return tmp

        for j in range(args.nsamples):
            handles = []
            for name in wrapped_layers_extra:
                handles.append(
                    subset_extra[name].register_forward_hook(
                        add_batch_extra(name, tars_extra[j])
                    )
                )

            with torch.no_grad():
                outs_extra[j] = layer_extra(
                    inps_extra[j].unsqueeze(0),
                    attention_mask=attention_mask_extra[j],
                    position_ids=position_ids_extra[j],
                )[0]

            for h in handles:
                h.remove()

        if not args.prune_part:
            for name in subset:
                print(f"pruning layer {i} name {name}")
                if args.use_diff or args.recover_from_base:
                    magnitude = torch.abs(
                        subset[name].weight.data - subset_base[name].weight.data
                    )
                else:
                    magnitude = torch.abs(subset[name].weight.data)
                # act = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))) - torch.sqrt(wrapped_layers_extra[name].scaler_row.reshape((1,-1)))
                act1 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                act2 = torch.sqrt(
                    wrapped_layers_extra[name].scaler_row.reshape((1, -1))
                )
                scale = torch.max(torch.sum(act1), torch.sum(act2))
                act1_norm = act1 / torch.sum(act1) * scale
                act2_norm = act2 / torch.sum(act2) * scale
                act = act1_norm - act2_norm
                W_metric = magnitude * act
                if args.neg_prune:
                    W_metric = -W_metric

                if args.dump_wanda_score:
                    if args.use_diff:
                        if args.disentangle:
                            save_folder = os.path.join(
                                args.save,
                                f"wanda_score/{prune_data}_utility_decouple_weight_diff_disentangle",
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_diff_disentangle.pkl",
                            )
                        else:
                            save_folder = os.path.join(
                                args.save,
                                f"wanda_score/{prune_data}_utility_decouple_weight_diff",
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_diff.pkl",
                            )
                    else:
                        if args.disentangle:
                            save_folder = os.path.join(
                                args.save,
                                f"wanda_score/{prune_data}_utility_decouple_weight_only_disentangle",
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_only_disentangle.pkl",
                            )
                        else:
                            save_folder = os.path.join(
                                args.save,
                                f"wanda_score/{prune_data}_utility_decouple_weight_only",
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_only.pkl",
                            )
                    with open(target_file, "wb") as f:
                        print(
                            "Writing W_metric in layer {} and name {} with {} to the file".format(
                                i, name, prune_data
                            )
                        )
                        pickle.dump(W_metric, f)
                    continue

                W_mask = (
                    torch.zeros_like(W_metric) == 1
                )  ## initialize a mask to be all False
                if prune_n != 0:
                    # structured n:m sparsity
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:, ii : (ii + prune_m)].float()
                            W_mask.scatter_(
                                1,
                                ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                                True,
                            )
                else:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)

                    if args.use_variant:
                        # wanda variant
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)

                        alpha = 0.4
                        alpha_hist = [0.0, 0.8]
                        W_mask, cur_sparsity = return_given_alpha(
                            alpha, sort_res, W_metric, tmp_metric, sum_before
                        )
                        while (
                            torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001
                        ) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                            if cur_sparsity > args.sparsity_ratio:
                                alpha_new = (alpha + alpha_hist[0]) / 2.0
                                alpha_hist[1] = alpha
                            else:
                                alpha_new = (alpha + alpha_hist[1]) / 2.0
                                alpha_hist[0] = alpha

                            alpha = alpha_new
                            W_mask, cur_sparsity = return_given_alpha(
                                alpha, sort_res, W_metric, tmp_metric, sum_before
                            )
                        print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                    else:
                        # unstructured pruning
                        indices = sort_res[1][
                            :, : int(W_metric.shape[1] * args.sparsity_ratio)
                        ]
                        W_mask.scatter_(1, indices, True)

                if args.recover_from_base:
                    assert model_base is not None
                    subset[name].weight.data[W_mask] = subset_base[name].weight.data[
                        W_mask
                    ]  # patch with the base model's weights
                else:
                    subset[name].weight.data[W_mask] = 0  ## set weights to zero
        else:
            # args.prune_part == True. We only prune the layer with low jaccard index, which is:
            # layer 0 mlp_down_proj
            # layer 1 self_attn._proj and mlp_down_proj
            # rest of layers: self_attn.o_proj, mlp_gate_proj, mlp_down_proj, mlp_up_proj
            for name in subset:
                condition = (
                    ((i == 0) and (name == "mlp.down_proj"))
                    or (
                        (i == 1)
                        and ((name == "self_attn.o_proj") or (name == "mlp.down_proj"))
                    )
                    or (
                        (i > 1)
                        and (
                            (name == "self_attn.o_proj")
                            or (name == "mlp.gate_proj")
                            or (name == "mlp.down_proj")
                            or (name == "mlp.up_proj")
                        )
                    )
                )
                if condition:
                    print(f"pruning layer {i} name {name}")
                    if args.use_diff or args.recover_from_base:
                        magnitude = torch.abs(
                            subset[name].weight.data - subset_base[name].weight.data
                        )
                    else:
                        magnitude = torch.abs(subset[name].weight.data)
                    # act = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1))) - torch.sqrt(wrapped_layers_extra[name].scaler_row.reshape((1,-1)))
                    act1 = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                    act2 = torch.sqrt(
                        wrapped_layers_extra[name].scaler_row.reshape((1, -1))
                    )
                    scale = torch.max(torch.sum(act1), torch.sum(act2))
                    act1_norm = act1 / torch.sum(act1) * scale
                    act2_norm = act2 / torch.sum(act2) * scale
                    W_metric = magnitude * act
                    if args.neg_prune:
                        W_metric = -W_metric

                    if args.dump_wanda_score:
                        # Only save the score, no pruning
                        save_folder = os.path.join(
                            args.save, f"wanda_score/{prune_data}__online"
                        )
                        if not os.path.exists(save_folder):
                            os.makedirs(save_folder)
                        if args.use_diff:
                            if args.disentangle:
                                save_folder = os.path.join(
                                    args.save,
                                    f"wanda_score/{prune_data}_utility_decouple_weight_diff_disentangle",
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_diff_disentangle.pkl",
                                )
                            else:
                                save_folder = os.path.join(
                                    args.save,
                                    f"wanda_score/{prune_data}_utility_decouple_weight_diff",
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_diff.pkl",
                                )
                        else:
                            if args.disentangle:
                                save_folder = os.path.join(
                                    args.save,
                                    f"wanda_score/{prune_data}_utility_decouple_weight_only_disentangle",
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_only_disentangle.pkl",
                                )
                            else:
                                save_folder = os.path.join(
                                    args.save,
                                    f"wanda_score/{prune_data}_utility_decouple_weight_only",
                                )
                                if not os.path.exists(save_folder):
                                    os.makedirs(save_folder)
                                target_file = os.path.join(
                                    save_folder,
                                    f"W_metric_layer_{i}_name_{name}_{prune_data}_utility_decouple_weight_only.pkl",
                                )
                        with open(target_file, "wb") as f:
                            print(
                                "Writing W_metric in layer {} and name {} with {} to the file".format(
                                    i, name, prune_data
                                )
                            )
                            pickle.dump(W_metric, f)
                        continue

                    W_mask = (
                        torch.zeros_like(W_metric) == 1
                    )  ## initialize a mask to be all False
                    if prune_n != 0:
                        # structured n:m sparsity
                        for ii in range(W_metric.shape[1]):
                            if ii % prune_m == 0:
                                tmp = W_metric[:, ii : (ii + prune_m)].float()
                                W_mask.scatter_(
                                    1,
                                    ii
                                    + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                                    True,
                                )
                    else:
                        sort_res = torch.sort(W_metric, dim=-1, stable=True)

                        if args.use_variant:
                            # wanda variant
                            tmp_metric = torch.cumsum(sort_res[0], dim=1)
                            sum_before = W_metric.sum(dim=1)

                            alpha = 0.4
                            alpha_hist = [0.0, 0.8]
                            W_mask, cur_sparsity = return_given_alpha(
                                alpha, sort_res, W_metric, tmp_metric, sum_before
                            )
                            while (
                                torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001
                            ) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                                if cur_sparsity > args.sparsity_ratio:
                                    alpha_new = (alpha + alpha_hist[0]) / 2.0
                                    alpha_hist[1] = alpha
                                else:
                                    alpha_new = (alpha + alpha_hist[1]) / 2.0
                                    alpha_hist[0] = alpha

                                alpha = alpha_new
                                W_mask, cur_sparsity = return_given_alpha(
                                    alpha, sort_res, W_metric, tmp_metric, sum_before
                                )
                            print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                        else:
                            # unstructured pruning
                            indices = sort_res[1][
                                :, : int(W_metric.shape[1] * args.sparsity_ratio)
                            ]
                            W_mask.scatter_(1, indices, True)

                    if args.recover_from_base:
                        assert model_base is not None
                        subset[name].weight.data[W_mask] = subset_base[
                            name
                        ].weight.data[
                            W_mask
                        ]  # patch with the base model's weights
                    else:
                        subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask[j],
                    position_ids=position_ids[j],
                )[0].squeeze(0)
            with torch.no_grad():
                outs_extra[j] = layer_extra(
                    inps_extra[j].unsqueeze(0),
                    attention_mask=attention_mask_extra[j],
                    position_ids=position_ids_extra[j],
                )[0].squeeze(0)
        inps, outs = outs, inps
        inps_extra, outs_extra = outs_extra, inps_extra

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_wandg_set_difference(
    args,
    model,
    tokenizer,
    model_base=None,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
    prune_data="align_short",
    p=0.5,
    q=0.5,
):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    if args.use_diff or args.recover_from_base:
        assert model_base is not None
        layers_base = model_base.model.layers
    metric1 = "alpaca_cleaned_no_safety"
    metric2 = prune_data

    print(
        "prune p = {}, q = {}, with metric1 = {}, metric2 = {}".format(
            p, q, metric1, metric2
        )
    )
    if args.prune_part:
        print("only prune the layer with low jaccard index")
    else:
        print("prune every linear layer")
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if args.use_diff or args.recover_from_base:
            subset_base = find_layers(layers_base[i])

        if not args.prune_part:
            for name in subset:
                print(f"pruning layer {i} name {name}")
                if args.model == "llama2-7b-chat-hf":
                    W_metric1 = pickle.load(
                        open(
                            f"out/llama2-7b-chat-hf/unstructured/wandg/{metric1}/wanda_score/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl",
                            "rb",
                        )
                    )
                    W_metric2 = pickle.load(
                        open(
                            f"out/llama2-7b-chat-hf/unstructured/wandg/{metric2}/wanda_score/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl",
                            "rb",
                        )
                    )
                elif args.model == "llama2-13b-chat-hf":
                    W_metric1 = pickle.load(
                        open(
                            f"out/llama2-13b-chat-hf/unstructured/wandg/{metric1}/wanda_score/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl",
                            "rb",
                        )
                    )
                    W_metric2 = pickle.load(
                        open(
                            f"out/llama2-13b-chat-hf/unstructured/wandg/{metric2}/wanda_score/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl",
                            "rb",
                        )
                    )
                else:
                    raise NotImplementedError

                top_p = int(
                    p * W_metric1.shape[1] * W_metric1.shape[0]
                )  # top_p utility
                top_q = int(q * W_metric2.shape[1] * W_metric2.shape[0])  # top_q safety

                top_p_indices = torch.topk(W_metric1.flatten(), top_p, largest=True)[1]
                top_q_indices = torch.topk(W_metric2.flatten(), top_q, largest=True)[1]
                unique_p = torch.unique(top_p_indices)
                unique_q = torch.unique(top_q_indices)

                # Create a boolean mask for elements in unique_q that are not in unique_p
                mask = ~torch.isin(unique_q, unique_p)

                # Apply the mask to unique_q to get filtered_indices
                filtered_indices = unique_q[mask]
                weight_dim = subset[name].weight.data.shape[1]
                filtered_indices_rows = filtered_indices // weight_dim
                filtered_indices_cols = filtered_indices % weight_dim

                assert (
                    args.dump_wanda_score == False
                )  # Only pruning from the saved score, won't save score again

                W_mask = torch.zeros_like(subset[name].weight.data) == 1
                W_mask[filtered_indices_rows, filtered_indices_cols] = (
                    True  # prune weights that has relatively high safety while not in top utility scores
                )

                if args.recover_from_base:
                    assert model_base is not None
                    subset[name].weight.data[W_mask] = subset_base[name].weight.data[
                        W_mask
                    ]  # patch with the base model's weights
                else:
                    subset[name].weight.data[W_mask] = 0  ## set weights to zero
        else:
            # args.prune_part == True. We only prune the layer with low jaccard index, which is:
            # layer 0 mlp_down_proj
            # layer 1 self_attn._proj and mlp_down_proj
            # rest of layers: self_attn.o_proj, mlp_gate_proj, mlp_down_proj, mlp_up_proj
            for name in subset:
                condition = (
                    ((i == 0) and (name == "mlp.down_proj"))
                    or (
                        (i == 1)
                        and ((name == "self_attn.o_proj") or (name == "mlp.down_proj"))
                    )
                    or (
                        (i > 1)
                        and (
                            (name == "self_attn.o_proj")
                            or (name == "mlp.gate_proj")
                            or (name == "mlp.down_proj")
                            or (name == "mlp.up_proj")
                        )
                    )
                )
                if condition:
                    W_metric1 = pickle.load(
                        open(
                            f"out/llama2-7b-chat-hf/unstructured/wandg/{metric1}/wanda_score/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl",
                            "rb",
                        )
                    )
                    W_metric2 = pickle.load(
                        open(
                            f"out/llama2-7b-chat-hf/unstructured/wandg/{metric2}/wanda_score/W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl",
                            "rb",
                        )
                    )
                    top_p = int(p * W_metric1.shape[1] * W_metric1.shape[0])
                    top_q = int(q * W_metric2.shape[1] * W_metric2.shape[0])

                    top_p_indices = torch.topk(
                        W_metric1.flatten(), top_p, largest=True
                    )[1]
                    top_q_indices = torch.topk(
                        W_metric2.flatten(), top_q, largest=True
                    )[1]
                    unique_p = torch.unique(top_p_indices)
                    unique_q = torch.unique(top_q_indices)

                    # Create a boolean mask for elements in unique_p that are not in unique_q
                    mask = ~torch.isin(unique_q, unique_p)

                    # Apply the mask to unique_p to get filtered_indices
                    filtered_indices = unique_q[mask]
                    weight_dim = subset[name].weight.data.shape[1]
                    filtered_indices_rows = filtered_indices // weight_dim
                    filtered_indices_cols = filtered_indices % weight_dim

                    assert (
                        args.dump_wanda_score == False
                    )  # Only pruning from the saved score, won't save score again

                    W_mask = torch.zeros_like(subset[name].weight.data) == 1
                    W_mask[filtered_indices_rows, filtered_indices_cols] = (
                        True  # prune weights that has relatively high safety while not in top utility scores
                    )

                    if args.recover_from_base:
                        assert model_base is not None
                        subset[name].weight.data[W_mask] = subset_base[
                            name
                        ].weight.data[
                            W_mask
                        ]  # patch with the base model's weights
                    else:
                        subset[name].weight.data[W_mask] = 0  ## set weights to zero

def prune_fluctuation_utility(
    args,
    model,
    tokenizer,
    model_base=None,
    model_extra=None,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
    prune_data="align",
):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    # load prune_data
    print(f"loading calibration data {prune_data}")
    dataloader, _ = get_loaders(
        prune_data,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
        disentangle=args.disentangle,
    )
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, tars, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, args.nsamples
        )

    if not args.disentangle:
        tars = [torch.zeros_like(tar) for tar in tars]  # remove -100's

    device_map_func = lambda tensor, device: tensor.to(device) if tensor is not None else None
    inps = [inp.squeeze(0).to(device) for inp in inps]
    tars = [tar.squeeze(0).to(device) for tar in tars]
    attention_mask = [device_map_func(am, device) for am in attention_mask]
    position_ids = [device_map_func(pids, device) for pids in position_ids]

    layers = model.model.layers
    if args.prune_part:
        print("only prune the layer with low jaccard index")
    else:
        print("prune every linear layer")

    attn_metric_list, mlp_metric_list = [], []
    attn_baseline_inp_list, mlp_baseline_inp_list = [], []
    attn_mask, mlp_mask = [], []

    for i in range(len(layers)):
        layer = layers[i]
        subset = {}
        subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        wrapped_layers = {}

        for name in subset:
            wrapped_layers[name] = BiasGPT(subset[name])

        # compute safety activation
        def add_batch(name, tar):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data, tar)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(
                subset[name].register_forward_hook(add_batch(name, tars[j]))
            )

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask[j],
                    position_ids=position_ids[j],
                )[0]

        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            structure = "AL-AM" # UL-UM or AL-AM
            metric = "WIFV"
            if name == 'self_attn.o_proj':
                W_metric = metrics[metric](wrapped_layers, subset, name) ** 2
                if structure == "UL-UM":
                    W_metric = W_metric.reshape(-1, 128).sum(dim=1)
                    thresh = torch.sort(W_metric.cuda())[0][int(args.sparsity_ratio*layer.self_attn.num_heads)].cpu()
                    W_mask = (W_metric>=thresh)
                    attn_mask.append(W_mask)
                else:
                    attn_metric_list.append(W_metric.cpu())
                attn_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
            else:
                W_metric = metrics[metric](wrapped_layers, subset, name)
                if structure == "UL-UM":
                    thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel()*args.sparsity_ratio)].cpu()
                    W_mask = (W_metric>=thresh)
                    mlp_mask.append(W_mask)
                else:
                    mlp_metric_list.append(W_metric.cpu())
                mlp_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))

            standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)
            if structure is "AL-AM":
                attn_metric = torch.stack(attn_metric_list)
                attn_metric = standarlization(attn_metric)
                attn_metric = attn_metric.reshape(len(layers), -1, 128).mean(dim=2)
                
                mlp_metric = torch.stack(mlp_metric_list)
                mlp_metric = standarlization(mlp_metric)
                
                prune_metric = torch.cat([attn_metric.view(-1), mlp_metric.view(-1)])
                sorted_prune, indices = torch.sort(prune_metric, descending=True)
                compression_weight = torch.ones_like(indices)
                compression_weight[indices < attn_metric.numel()] = 512.0 / 3
                threshold = sorted_prune[torch.argmin(torch.abs(torch.cumsum(compression_weight, 0) - torch.sum(compression_weight)*(1 - args.pruning_ratio)))]
                attn_mask = (attn_metric > threshold)
                mlp_mask = (mlp_metric > threshold)

                # For components that are redundant for utility, we set mask value to 0, others to 1
                # Save mask for the later statistics work
            else:
                attn_mask = torch.stack(attn_mask) 
                mlp_mask = torch.stack(mlp_mask)

        for idx in range(len(layers)):
            compress(model.model.layers[idx], attn_mask[idx], None, attn_baseline_inp_list[idx], None, device, unstr=args.unstr)
            compress(model.model.layers[idx], None, mlp_mask[idx], None, mlp_baseline_inp_list[idx], device, unstr=args.unstr)

        # for j in range(args.nsamples):
        #     with torch.no_grad():
        #         outs[j] = layer(
        #             inps[j].unsqueeze(0),
        #             attention_mask=attention_mask[j],
        #             position_ids=position_ids[j],
        #         )[0].squeeze(0)
        #     with torch.no_grad():
        #         outs_extra[j] = layer_extra(
        #             inps_extra[j].unsqueeze(0),
        #             attention_mask=attention_mask_extra[j],
        #             position_ids=position_ids_extra[j],
        #         )[0].squeeze(0)
        inps, outs = outs, inps
        inps_extra, outs_extra = outs_extra, inps_extra

    model.config.use_cache = use_cache
    torch.cuda.empty_cache() 

def prune_fluctuation_decouple_utility_and_safety(
    args,
    model,
    tokenizer,
    model_base=None,
    model_extra=None,
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
    prune_data="align",
):
    """
    Compute wanda score based on the difference between tow activation norms (In an online way, do not need to load wanda score from file)

    Compute the norms first then compute the difference

    Args:
        model_extra (_type_, optional): An extra model to compute utility activation. For the consideration of the interference in KV cache, currently we are using to models to compute the safety and utility activation respectively. Defaults to None.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if args.decouple_align_utility:
        prune_data_extra = "alpaca_cleaned_no_safety"
    elif args.decouple_align_misalign:
        prune_data_extra = "misalign"
    else:
        raise NotImplementedError
    # load prune_data
    print(f"loading calibration data {prune_data}")
    dataloader, _ = get_loaders(
        prune_data,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
        disentangle=args.disentangle,
    )
    print("dataset loading complete")
    print(f"loading extra calibration data {prune_data_extra}")
    dataloader_extra, _ = get_loaders(
        prune_data_extra,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model_extra.seqlen,
        tokenizer=tokenizer,
        disentangle=args.disentangle,
    )
    print("extra dataset loading complete")
    with torch.no_grad():
        inps, outs, tars, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, args.nsamples
        )
    with torch.no_grad():
        inps_extra, outs_extra, tars_extra, attention_mask_extra, position_ids_extra = (
            prepare_calibration_input(
                model_extra, dataloader_extra, device, args.nsamples
            )
        )

    if not args.disentangle:
        tars = [torch.zeros_like(tar) for tar in tars]  # remove -100's
        tars_extra = [torch.zeros_like(tar) for tar in tars_extra]  # remove -100's

    device_map_func = lambda tensor, device: tensor.to(device) if tensor is not None else None
    inps = [inp.squeeze(0).to(device) for inp in inps]
    inps_extra = [inp.squeeze(0).to(device) for inp in inps_extra]
    tars = [tar.squeeze(0).to(device) for tar in tars]
    tars_extra = [tar.squeeze(0).to(device) for tar in tars_extra]
    attention_mask = [device_map_func(am, device) for am in attention_mask]
    attention_mask_extra = [device_map_func(am, device) for am in attention_mask_extra]
    position_ids = [device_map_func(pids, device) for pids in position_ids]
    position_ids_extra = [device_map_func(pids, device) for pids in position_ids_extra]

    layers = model.model.layers
    layers_extra = model_extra.model.layers

    if args.prune_part:
        print("only prune the layer with low jaccard index")
    else:
        print("prune every linear layer")

    attn_metric_list, mlp_metric_list = [], []
    attn_baseline_inp_list, mlp_baseline_inp_list = [], []
    attn_mask, mlp_mask = [], []

    extra_attn_metric_list, extra_mlp_metric_list = [], []
    extra_attn_baseline_inp_list, extra_mlp_baseline_inp_list = [], []
    extra_attn_mask, extra_mlp_mask = [], []

    for i in range(len(layers)):
        layer = layers[i]
        layer_extra = layers_extra[i]
        subset = {}
        subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        subset_extra = {}
        subset_extra.update({'self_attn.o_proj': find_layers(layer_extra)['self_attn.o_proj']})
        subset_extra.update({'mlp.down_proj': find_layers(layer_extra)['mlp.down_proj']})

        wrapped_layers = {}
        wrapped_layers_extra = {}

        for name in subset:
            wrapped_layers[name] = BiasGPT(subset[name])
            wrapped_layers_extra[name] = BiasGPT(subset_extra[name])

        # compute safety activation
        def add_batch(name, tar):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data, tar)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(
                subset[name].register_forward_hook(add_batch(name, tars[j]))
            )

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask[j],
                    position_ids=position_ids[j],
                )[0]

        for h in handles:
            h.remove()

        # compute utility activation
        def add_batch_extra(name, tar):
            def tmp(_, inp, out):
                wrapped_layers_extra[name].add_batch(inp[0].data, out.data, tar)
            return tmp

        handles = []
        for name in wrapped_layers_extra:
            handles.append(
                subset_extra[name].register_forward_hook(
                    add_batch_extra(name, tars_extra[j])
                )
            )

        for j in range(args.nsamples):
            with torch.no_grad():
                outs_extra[j] = layer_extra(
                    inps_extra[j].unsqueeze(0),
                    attention_mask=attention_mask_extra[j],
                    position_ids=position_ids_extra[j],
                )[0]

        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            structure = "AL-AM" # UL-UM or AL-AM
            metric = "WIFV"
            if name == 'self_attn.o_proj':
                W_metric = metrics[metric](wrapped_layers, subset, name) ** 2
                W_metric_extra = metrics[metric](wrapped_layers_extra, subset_extra, name) ** 2
                if structure == "UL-UM":
                    # W_metric = W_metric.reshape(-1, 128).sum(dim=1)
                    # thresh = torch.sort(W_metric.cuda())[0][int(args.sparsity_ratio*layer.self_attn.num_heads)].cpu()
                    # W_mask = (W_metric>=thresh)
                    # attn_mask.append(W_mask)
                    pass
                else:
                    attn_metric_list.append(W_metric.cpu())
                    extra_attn_metric_list.append(W_metric_extra.cpu())
                attn_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
                extra_attn_baseline_inp_list.append(wrapped_layers_extra[name].baseline_inp.type(torch.half))
            else:
                W_metric = metrics[metric](wrapped_layers, subset, name)
                W_metric_extra = metrics[metric](wrapped_layers_extra, subset_extra, name)
                if structure == "UL-UM":
                    # thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel()*args.sparsity_ratio)].cpu()
                    # W_mask = (W_metric>=thresh)
                    # mlp_mask.append(W_mask)
                    pass
                else:
                    mlp_metric_list.append(W_metric.cpu())
                    extra_mlp_metric_list.append(W_metric_extra.cpu())
                mlp_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
                extra_mlp_baseline_inp_list.append(wrapped_layers_extra[name].baseline_inp.type(torch.half))

            standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)
            if structure is "AL-AM":
                attn_metric = torch.stack(attn_metric_list)
                attn_metric = standarlization(attn_metric)
                attn_metric = attn_metric.reshape(len(layers), -1, 128).mean(dim=2)
                
                mlp_metric = torch.stack(mlp_metric_list)
                mlp_metric = standarlization(mlp_metric)
                prune_metric = torch.cat([attn_metric.view(-1), mlp_metric.view(-1)])
                
                # sorted_prune, indices = torch.sort(prune_metric, descending=True)
                # compression_weight = torch.ones_like(indices)
                # compression_weight[indices < attn_metric.numel()] = 512.0 / 3
                # threshold = sorted_prune[torch.argmin(torch.abs(torch.cumsum(compression_weight, 0) - torch.sum(compression_weight)*(1 - args.sparsity_ratio)))]
                # attn_mask = (attn_metric > threshold)
                # mlp_mask = (mlp_metric > threshold)

                attn_metric_extra = torch.stack(extra_attn_metric_list)
                attn_metric_extra = standarlization(attn_metric_extra)
                attn_metric_extra = attn_metric_extra.reshape(len(layers), -1, 128).mean(dim=2)
                
                mlp_metric_extra = torch.stack(extra_mlp_metric_list)
                mlp_metric_extra = standarlization(mlp_metric_extra)
                prune_metric_extra = torch.cat([attn_metric_extra.view(-1), mlp_metric_extra.view(-1)])

                sum_metric = prune_metric + prune_metric_extra
                diff_metric = prune_metric_extra - prune_metric

                attn_mask = torch.zeros_like(attn_metric)
                mlp_mask = torch.zeros_like(mlp_metric)

                # we first use sum metric to confirm compoenets that are redundant for both utility and safety, how the percent, we set mask value to 0

                # we then use diff metric to remove largest value to confirm components that contribute for safety, we set mask value to 1 

                # we then use diff metric to remove smallest value to confirm components that contribute for utility, we set mask value to 2  

                # exclude the above components will contribute to both utility and safety, we set mask value to 3

            else:
                pass
                # attn_mask = torch.stack(attn_mask) 
                # mlp_mask = torch.stack(mlp_mask)

        for idx in range(len(layers)):
            compress(model.model.layers[idx], attn_mask[idx], None, attn_baseline_inp_list[idx], None, device, unstr=args.unstr)
            compress(model.model.layers[idx], None, mlp_mask[idx], None, mlp_baseline_inp_list[idx], device, unstr=args.unstr)

        # for j in range(args.nsamples):
        #     with torch.no_grad():
        #         outs[j] = layer(
        #             inps[j].unsqueeze(0),
        #             attention_mask=attention_mask[j],
        #             position_ids=position_ids[j],
        #         )[0].squeeze(0)
        #     with torch.no_grad():
        #         outs_extra[j] = layer_extra(
        #             inps_extra[j].unsqueeze(0),
        #             attention_mask=attention_mask_extra[j],
        #             position_ids=position_ids_extra[j],
        #         )[0].squeeze(0)
        inps, outs = outs, inps
        inps_extra, outs_extra = outs_extra, inps_extra

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print("Starting ...")
    dataloader, _ = get_loaders(
        "wikitext2",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    # dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    print("Ready.")

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print("Pruning ...")

            gpts[name].fasterprune(
                args.sparsity_ratio,
                prune_n=prune_n,
                prune_m=prune_m,
                percdamp=0.01,
                blocksize=128,
            )
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print("Starting ...")
    dataloader, _ = get_loaders(
        "wikitext2",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    # dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    print("Ready.")

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print("Pruning ...")

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(
                    args.sparsity_ratio, prune_n, prune_m
                )
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(
                    args.sparsity_ratio, prune_n, prune_m
                )
            elif "iter" in args.prune_method:
                prune_mask = None

            gpts[name].fasterprune(
                args,
                args.sparsity_ratio,
                mask=prune_mask,
                prune_n=prune_n,
                prune_m=prune_m,
                percdamp=0.01,
                blocksize=128,
            )
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_attention_head(
    args, model, model_base=None, device=torch.device("cuda:0"), top_k_heads=10
):
    """Prune the attention_heads based on the probing results. Still not supporting reover from base. Only support Llama-2-7b-chat-hf

    Args:
        args (_type_): _description_
        model (_type_): _description_
        model_base (_type_, optional): _description_. Defaults to None.
        device (_type_, optional): _description_. Defaults to torch.device("cuda:0").

    Raises:
        ValueError: _description_
    """

    layers = model.model.layers
    k = top_k_heads
    print("Pruning top {} attention heads".format(k))

    # find the top-k attention heads in probing results based on the value in the probing_result
    if args.model == "llama2-7b-chat-hf":
        with open("data/probing_result_7b.json", "r") as f:
            # read json file to dict
            probing_result = json.load(f)
        count = sum(value == 1.0 for value in probing_result.values())
        if k <= count:
            top_k_heads_full = heapq.nlargest(
                132, probing_result, key=probing_result.get
            )
            top_k_heads = random.sample(top_k_heads_full, k)
        elif k <= len(probing_result):
            top_k_heads = heapq.nlargest(k, probing_result, key=probing_result.get)
        else:
            raise ValueError("k is larger than the number of attention heads")

        extracted_numbers = [
            list(map(int, re.findall(r"\d+", head))) for head in top_k_heads
        ]

        for head in extracted_numbers:
            block_id = head[0]
            head_id = head[1]
            layer = layers[block_id]
            subset = find_layers(layer)
            for name in ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"]:
                W = subset[name].weight.data
                W_metric = torch.zeros_like(W)
                W_metric[:, head_id * 128 : (head_id + 1) * 128] = 1
                W_mask = W_metric == 1
                subset[name].weight.data[W_mask] = 0
            name = "self_attn.o_proj"
            W = subset[name].weight.data
            W_metric = torch.zeros_like(W)
            W_metric[head_id * 128 : (head_id + 1) * 128, :] = 1
            W_mask = W_metric == 1
            subset[name].weight.data[W_mask] = 0
    elif args.model == "llama2-13b-chat-hf":
        with open("data/probing_result_13b.json", "r") as f:
            # read json file to dict
            probing_result = json.load(f)
        count = sum(value == 1.0 for value in probing_result.values())
        if k <= count:
            top_k_heads_full = heapq.nlargest(
                count, probing_result, key=probing_result.get
            )
            top_k_heads = random.sample(top_k_heads_full, k)
        elif k <= len(probing_result):
            top_k_heads = heapq.nlargest(k, probing_result, key=probing_result.get)
        else:
            raise ValueError("k is larger than the number of attention heads")

        extracted_numbers = [
            list(map(int, re.findall(r"\d+", head))) for head in top_k_heads
        ]

        for head in extracted_numbers:
            block_id = head[0]
            head_id = head[1]
            layer = layers[block_id]
            subset = find_layers(layer)
            for name in ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"]:
                W = subset[name].weight.data
                head_dim = W.shape[1] // 40
                W_metric = torch.zeros_like(W)
                W_metric[:, head_id * head_dim : (head_id + 1) * head_dim] = 1
                W_mask = W_metric == 1
                subset[name].weight.data[W_mask] = 0
            name = "self_attn.o_proj"
            W = subset[name].weight.data
            W_metric = torch.zeros_like(W)
            W_metric[head_id * head_dim : (head_id + 1) * head_dim, :] = 1
            W_mask = W_metric == 1
            subset[name].weight.data[W_mask] = 0
