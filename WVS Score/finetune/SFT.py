import torch
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

base_model = 'meta-llama/Llama-3.2-1B-Instruct'
dataset = load_dataset("json", data_files="../../data/WVS/WVQ_China_Train.jsonl", split="train")

max_memory = {i: '46000MB' for i in range(torch.cuda.device_count())}
model = LlamaForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    max_memory=max_memory
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

training_params = TrainingArguments(
    output_dir="./SFT-fine-tune-results",
    num_train_epochs=6,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    label_names=[str(i) for i in range(0, 11)]
)

peft_params = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    tokenizer=tokenizer,
    args=training_params
)

print("Fine-tuning model...")
trainer.train()

output_dir = "../models/SFT-LoRA-Llama-3.2-1B-Instruct"
trainer.model.save_pretrained(output_dir)
trainer.tokenizer.save_pretrained(output_dir)

print("Fine-tuning complete! Model saved to", output_dir)