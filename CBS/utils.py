import torch
import pandas as pd
import numpy as np
import os

from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    # LlamaTokenizer,
    # LlamaConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    # pipeline,
    # logging,
)
from peft import PeftModel

def load_model(model_name_or_path, casual_lm=AutoModelForCausalLM):
    compute_dtype = getattr(torch, "bfloat16")
    # quant_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=compute_dtype,
    #     bnb_4bit_use_double_quant=False,
    # )

    num_devices = torch.cuda.device_count()
    total_vram = [torch.cuda.get_device_properties(i).total_memory / (1024 ** 2) for i in range(num_devices)]  # In MB
    max_memory = {i: f"{int(vram * 0.9)}MB" for i, vram in enumerate(total_vram)}
    # print(f"Max Memory Config: {max_memory}")

    model = casual_lm.from_pretrained(
        model_name_or_path,
        # quantization_config=quant_config,
        torch_dtype=compute_dtype,
        device_map="auto",
        max_memory=max_memory
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    # print(f"Loaded base model at {model_name_or_path}")

    return model

def merge_model_with_adapter(model, adapter_path):
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    # print(f"Merged adapter at {adapter_path}")
    

def load_model_and_tokenizer(model_name_or_path, adapter_path="", casual_lm=AutoModelForCausalLM):
    model = load_model(model_name_or_path, casual_lm)

    if adapter_path != "":
        merge_model_with_adapter(model, adapter_path)
        # print(f"Adapater at '{adapter_path}' loaded")
    # else:
        # print(f"Model {model_name_or_path} loaded")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer