model="llama2-7b-chat-hf"
method="wandg_set_difference"
type="unstructured"
suffix="weightonly"
data="align_short"
save_dir="out/$model/$type/${method}/${data}/"

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model $model \
    --prune_method $method \
    --sparsity_ratio 0.5 \
    --prune_data $data \
    --p 0.1\
    --q 0.1\
    --sparsity_type $type \
    --save $save_dir \
    --eval_zero_shot \
    --eval_attack \
    --save_attack_res