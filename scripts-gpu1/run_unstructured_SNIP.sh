model="llama2-7b-chat-hf"
# model="llama2-7b-hf"
method="wandg"
# data="alpaca_cleaned_no_safety"
data="align_short"
type="unstructured"
save_dir="out1/$model/$type/${method}/${data}/"

CUDA_VISIBLE_DEVICES=1 python main.py \
    --model $model \
    --prune_method $method \
    --prune_data $data \
    --swap_space 42 \
    --temp_dir .temp1 \
    --sparsity_ratio 0.5 \
    --sparsity_type $type \
    --neg_prune \
    --save $save_dir \
    --dump_wanda_score
    # --eval_zero_shot \
    # --eval_attack \
    # --save_attack_res \
    # --dump_wanda_score