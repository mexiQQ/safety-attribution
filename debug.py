from vllm import LLM

vllm_model = LLM(
    model=".temp/wanda_usediff_False_recover_False", 
    tokenizer="meta-llama/Llama-2-7b-hf", 
    dtype="bfloat16", 
    gpu_memory_utilization=0.75,
    swap_space=128
)

import pdb; pdb.set_trace()