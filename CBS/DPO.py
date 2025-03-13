import torch
import torch.nn.functional as F
import json
import os

from trl import DPOConfig, DPOTrainer
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from datasets import Dataset

import config
import utils

# Load JSON dataset
def load_dpo_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    dataset = {"prompt": [], "chosen": [], "rejected": []}
    for entry in data:
        dataset["prompt"].append(entry[0])
        dataset["chosen"].append(entry[1])
        dataset["rejected"].append(entry[2])
    
    return Dataset.from_dict(dataset)

def dpo_finetune(config: config.DPOConfig, sft_model, sft_tokenizer):
    if not os.path.exists(config.dpo_adapter_save_path):
        os.makedirs(config.dpo_adapter_save_path)
        print(f"Directory {config.dpo_adapter_save_path} created")

    dataset = load_dpo_dataset(config.dpo_dataset_path)
    print("Loaded DPO dataset")

    training_args = config.training_args

    trainer = config.trainer(
        sft_model,
        args=config.training_args,
        train_dataset=dataset,
        processing_class=sft_tokenizer,
        peft_config=config.peft_config,
    )

    trainer.train()

    trainer.model.save_pretrained(config.dpo_adapter_save_path)
    trainer.processing_class.save_pretrained(config.dpo_adapter_save_path)
    print(f"Model and tokenizer saved to {config.dpo_adapter_save_path}")


def main():
    args = config.Args()
    dpo_config = config.DPOConfig(args)

    sft_model, sft_tokenizer = utils.load_model_and_tokenizer(dpo_config.huggingface_model_name,
                                                              adapter_path=dpo_config.sft_model_adapter_path,
                                                              casual_lm=dpo_config.casual_lm)
    dpo_finetune(dpo_config, sft_model, sft_tokenizer)
    

if __name__ == "__main__":
    main()
