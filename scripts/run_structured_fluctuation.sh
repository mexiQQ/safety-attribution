model="llama2-7b-chat-hf"
method="fluctuation"
type="structured"
suffix="weightonly"
data="alpaca_cleaned_no_safety"
save_dir="out/$model/$type/${method}/${data}/"

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model $model \
    --prune_method $method \
    --prune_data align \
    --sparsity_ratio 0.2 \
    --sparsity_type $type \
    --neg_prune \
    --save $save_dir \
    --eval_zero_shot \
    # --eval_attack \
    # --save_attack_res \
    # --decouple_align_utility