model="llama2-7b-chat-hf"
method="fluctuation"
type="structured"
data="alpaca_cleaned_no_safety"
save_dir="out2/$model/$type/${method}/${data}/"

CUDA_VISIBLE_DEVICES=2 python main.py \
    --model $model \
    --prune_method $method \
    --prune_data $data \
    --sparsity_ratio 0.1 \
    --swap_space 42 \
    --temp_dir .temp2 \
    --sparsity_type $type \
    --neg_prune \
    --save $save_dir \
    --eval_zero_shot \
    # --eval_attack \
    # --save_attack_res \
    # --decouple_align_utility