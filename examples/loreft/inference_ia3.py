import transformers
import numpy as np
import torch
import time
from tqdm import tqdm
import fire

from src_inference.peft import TaskType, IA3Config, get_peft_model

def main(model_size="7b", bs=8):
    if model_size == "13b":
        model_name_or_path = "yahma/llama-13b-hf"
    else:
        model_name_or_path = "yahma/llama-7b-hf"

    device = "cuda"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        torch_dtype=torch.bfloat16, 
        device_map=device
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=False
    )

    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ]
    peft_config = IA3Config(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,
        target_modules=target_modules,
        init_ia3_weights=True,
    )
    model = get_peft_model(model, peft_config)
    print(model)

    inputs = tokenizer([""] * bs, return_tensors="pt").to(device)
    print("Input size:", inputs.input_ids.size())

    samples = 16
    max_new_tokens = 1024
    start = time.time()
    for i in tqdm(range(samples)):
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    end = time.time()
    print(max_new_tokens * samples * bs / (end - start)) 


if __name__ == "__main__":
    fire.Fire(main)