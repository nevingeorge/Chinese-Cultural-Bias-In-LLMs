import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig
from trl import SFTTrainer
import evaluate_WVS_score

base_model = 'meta-llama/Llama-3.2-1B-Instruct'
dataset = load_dataset("json", data_files="../../data/WVS/WVQ_China_Train.jsonl", split="train")

# The first config is the one used in CultureLLM
lora_configurations = [{"lora_alpha": 16, "lora_dropout": 0.1, "r": 64}, 
                       {"lora_alpha": 16, "lora_dropout": 0.1, "r": 8}, 
                       {"lora_alpha": 32, "lora_dropout": 0.1, "r": 16}, 
                       {"lora_alpha": 64, "lora_dropout": 0.05, "r": 32}, 
                       {"lora_alpha": 128, "lora_dropout": 0.0, "r": 64}]

results = []

for lora_config in lora_configurations:
    print("Fine-tuning model with LoRA hyperparameters:", lora_config)

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

    peft_params = LoraConfig(
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        r=lora_config["r"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_params = TrainingArguments(
        output_dir="./SFT-fine-tune-results",
        num_train_epochs=6,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="none",
        save_strategy="no",
        logging_dir=None,
        label_names=[str(i) for i in range(0, 11)]
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

    print("Finished fine-tuning. Beginning evaluation...")

    model = trainer.model.merge_and_unload()
    tokenizer = trainer.tokenizer

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    final_score, total = evaluate_WVS_score.obtain_results("../../data/WVS/WVQ_China_Evaluate.jsonl", pipe)

    results.append((lora_config, final_score, total))

with open("results_SFT_LoRA_hyperparameter_search.txt", "w") as file:
    for result in results:
        lora_config, final_score, total = result
        file.write(f"LoRA hyperparameters: {lora_config}\n")
        file.write(f"WVS score: {final_score:.2f}%\n")
        file.write(f"Total valid responses: {total}\n")
        file.write("\n")

print("Finished LoRA hyperparameter search. Results are saved in results_SFT_LoRA_hyperparameter_search.txt.")