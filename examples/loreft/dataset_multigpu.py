from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
from typing import Dict

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import transformers

from task_config import task_config
from templates import *



class SupervisedDataset(Dataset):
    def __init__(
        self, 
        task: str, 
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_split="train", 
        dataset=None, 
        seed=42, 
        max_n_example=None, 
        **kwargs,
    ):
        super(SupervisedDataset, self).__init__()
        
        result = defaultdict(list)
        self.raw_dataset, self.trigger_tokens, self.num_labels = None, None, None
        
        dataset_config = task_config[task]
        task_prompt_template = dataset_config["task_prompt_template"]
        trigger_tokens = dataset_config["trigger_tokens"]
        self.trigger_tokens = trigger_tokens

        if dataset is None:
            print("loading data for dataset: ", data_path)
            if task in ["alpaca", "instruct", "ultrafeedback"] and data_split != "train":
                task_dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
            elif data_path.endswith(".json"):
                task_dataset = load_dataset("json", data_files=data_path)[data_split]
            else:
                task_dataset = load_dataset(data_path)[data_split]
        if max_n_example is not None:
            task_dataset = task_dataset.shuffle(seed=seed)
            task_dataset = task_dataset.select(range(max_n_example))

        # save raw_dataset pointer for access raw strings
        self.raw_dataset = task_dataset if data_split != "train" else None

        # tokenize and intervene
        for i, data_item in enumerate(tqdm(task_dataset)):
            # set up prompt
            if task == "commonsense":
                base_prompt = task_prompt_template % (data_item['instruction'])
                base_input = base_prompt + trigger_tokens + data_item["answer"] + tokenizer.eos_token
            elif task == "math": # we strip since these are model generated examples.
                base_prompt = task_prompt_template % (data_item['instruction'])
                base_input = base_prompt + data_item["output"] + tokenizer.eos_token
            elif task == "alpaca" or task == "instruct" or task == "ultrafeedback":
                if 'input' not in data_item or data_item['input'] == "":
                    base_prompt = alpaca_prompt_no_input_template % (data_item['instruction'])
                else:
                    base_prompt = task_prompt_template % (data_item['instruction'], data_item['input'])
                base_input = base_prompt + data_item["output"] + tokenizer.eos_token
            elif task == "gsm8k": # setup is from https://github.com/yxli2123/LoftQ/
                base_prompt = task_prompt_template % (
                    "Answer the above question. First think step by step and then answer the final number.",
                    data_item['question']
                )
                base_input = base_prompt + data_item["answer"].replace("####", "The final answer is: ") + \
                    tokenizer.eos_token
            else:
                raise ValueError(f"Unrecognized task: {task}")
            
            # tokenize
            base_prompt_ids = tokenizer(
                base_prompt, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
            base_prompt_length = len(base_prompt_ids)
            if data_split == "train":
                base_input_ids = tokenizer(
                    base_input, max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")["input_ids"][0]
                output_ids = deepcopy(base_input_ids)
                output_ids[:base_prompt_length] = IGNORE_INDEX
                    
                result["input_ids"].append(base_input_ids)
                result["labels"].append(output_ids)
            else:
                result["input_ids"].append(base_prompt_ids)
                
            result["id"].append(i)
            
            # add a single padding token BEFORE input_ids and fix everything
            result["input_ids"][-1] = torch.cat((torch.tensor([tokenizer.pad_token_id,]), result["input_ids"][-1]))
            if data_split == "train":
                result["labels"][-1] = torch.cat((torch.tensor([IGNORE_INDEX]), result["labels"][-1]))
            result["attention_mask"].append((result["input_ids"][-1] != tokenizer.pad_token_id).int())

        self.input_ids = result["input_ids"]
        self.attention_mask = result["attention_mask"]
        self.labels = result["labels"] if "labels" in result else None
        self.id = result["id"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.labels is not None:
            return dict(
                input_ids=self.input_ids[i],
                attention_mask=self.attention_mask[i],
                labels=self.labels[i],
            )
        else:
            return dict(
                input_ids=self.input_ids[i],
                attention_mask=self.attention_mask[i],
                id=self.id[i],
            )