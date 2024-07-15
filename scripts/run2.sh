model="llama2-7b-chat-hf"
method="wanda"
type="unstructured"
suffix="weightonly"
save_dir="out/$model/$type/${method}_${suffix}/align/"

CUDA_VISIBLE_DEVICES=0 python main_transfer.py \
    --model $model \
    --prune_method $method \
    --prune_data align \
    --sparsity_ratio 0.5 \
    --sparsity_type $type \
    --neg_prune \
    --save $save_dir \
    --eval_zero_shot \
    --eval_attack \
    --save_attack_res \
    --decouple_align_utility