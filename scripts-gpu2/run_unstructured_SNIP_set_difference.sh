model="llama2-7b-chat-hf"
method="wandg_set_difference"
type="unstructured"
data="align_short"
save_dir="out2/$model/$type/${method}/${data}/"

CUDA_VISIBLE_DEVICES=2 python main.py \
    --model $model \
    --prune_method $method \
    --sparsity_ratio 0.5 \
    --prune_data $data \
    --swap_space 42 \
    --temp_dir .temp2 \
    --p 0.1\
    --q 0.1\
    --sparsity_type $type \
    --save $save_dir \
    --eval_zero_shot \
    --eval_attack \
    --save_attack_res