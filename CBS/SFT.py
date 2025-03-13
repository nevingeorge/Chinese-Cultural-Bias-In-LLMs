import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

from config import Args, SFTConfig
import utils

def sft_finetune(config: SFTConfig):
    if not os.path.exists(config.sft_adapter_save_path):
        os.makedirs(config.sft_adapter_save_path)
        print(f"Directory {config.sft_adapter_save_path} created")

    train_dataset = load_dataset("json", data_files=config.sft_dataset_path, split="train")

    base_model, tokenizer = utils.load_model_and_tokenizer(config.huggingface_model_name, 
                                                           adapter_path=config.adapter_path_to_load,
                                                           casual_lm=config.casual_lm)

    training_args = config.training_args

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        peft_config=config.peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    trainer.train()

    trainer.model.save_pretrained(config.sft_adapter_save_path)
    trainer.processing_class.save_pretrained(config.sft_adapter_save_path)
    print(f"Model and tokenizer saved to {config.sft_adapter_save_path}")


def get_sft_model_and_tokenizer(config: SFTConfig):
    if config.existing_sft_adapter_path != "":
        # load existing sft model
        if not os.path.exists(config.existing_sft_adapter_path):
            raise ValueError(f"SFT adapter path {config.existing_sft_adapter_path} not existed.")
        model, tokenizer = utils.load_model_and_tokenizer(config.huggingface_model_name,
                                                          adapter_path=config.existing_sft_adapter_path,
                                                          casual_lm=config.casual_lm
                                                          )
    else:
        # finetune a sft model
        sft_finetune(config)
        model, tokenizer = utils.load_model_and_tokenizer(config.huggingface_model_name,
                                                          adapter_path=config.sft_adapter_save_path,
                                                          casual_lm=config.casual_lm
                                                          )
    return model, tokenizer

def main():
    args = Args()
    sft_config = SFTConfig(args)
    sft_finetune(sft_config)

if __name__ == "__main__":
    main()
