model="llama2-7b-chat-hf"
method="fluctuation_set_difference"
type="structured"
data="align"
save_dir="out2/$model/$type/${method}/${data}/"

CUDA_VISIBLE_DEVICES=2 python main.py \
    --model $model \
    --prune_method $method \
    --prune_data $data \
    --sparsity_ratio 0.01 \
    --fluctuation_case 4 \
    --fluctuation_case_four_hyper 0.1 0.01 0.1 \
    --swap_space 128 \
    --temp_dir .temp2 \
    --sparsity_type $type \
    --neg_prune \
    --save $save_dir \
    --eval_zero_shot \
    --decouple_align_utility \
    --eval_attack \
    --save_attack_res \
    # --decouple_align_utility