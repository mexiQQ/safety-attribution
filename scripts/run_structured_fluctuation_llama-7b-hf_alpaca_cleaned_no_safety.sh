model="llama-7B-hf"
method="fluctuation"
type="structured"
data="alpaca_cleaned_no_safety"
save_dir="out/$model/$type/${method}/${data}/"

CUDA_VISIBLE_DEVICES=2 python main.py \
    --model $model \
    --prune_method $method \
    --prune_data $data \
    --sparsity_ratio 0.1 \
    --fluctuation_case 4 \
    --swap_space 42 \
    --temp_dir .temp \
    --sparsity_type $type \
    --neg_prune \
    --save $save_dir \
    --save_model "$save_dir/weight/sparsity_10" \
    # --eval_zero_shot \
    # --eval_attack \
    # --save_attack_res \
    # --decouple_align_utility