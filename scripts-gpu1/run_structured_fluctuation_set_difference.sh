model="llama2-7b-chat-hf"
method="fluctuation_set_difference"
type="structured"
data="align"
save_dir="out1/$model/$type/${method}/${data}/"

CUDA_VISIBLE_DEVICES=1 python main.py \
    --model $model \
    --prune_method $method \
    --prune_data $data \
    --sparsity_ratio 0.2 \
    --swap_space 42 \
    --temp_dir .temp1 \
    --sparsity_type $type \
    --neg_prune \
    --save $save_dir \
    --eval_zero_shot \
    --decouple_align_utility \
    # --eval_attack \
    # --save_attack_res \
    # --decouple_align_utility