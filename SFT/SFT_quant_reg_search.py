import torch
import torch.nn.functional as F
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
import SAR

base_model = 'meta-llama/Llama-3.2-1B-Instruct'
dataset = load_dataset("json", data_files="../data/WVQ_China_Train.jsonl", split="train")

# quantization, SAR
options = [[0,0], [1,0], [0,1], [1,1]]

with open("results_SFT_quant_reg_search.txt", "w") as file:
    for option in options:
        print("Option (quantization, SAR):", option)
        file.write(f"Setting (quantization, SAR): {option}\n")

        max_memory = {i: '46000MB' for i in range(torch.cuda.device_count())}

        if option[0]: # use quantization
            compute_dtype = getattr(torch, "float16")

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=False,
            )

            model = LlamaForCausalLM.from_pretrained(
                base_model,
                quantization_config=quant_config,
                device_map="auto",
                max_memory=max_memory
            )
        else:
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

        peft_params = LoraConfig(
            lora_alpha=32,
            lora_dropout=0.1,
            r=16,
            bias="none",
            task_type="CAUSAL_LM",
        )

        if option[1]: # Use SAR
            trainer = SAR.SARTrainer(
                model=model,
                train_dataset=dataset,
                peft_config=peft_params,
                tokenizer=tokenizer,
                args=training_params,
                epsilon=1e-3,   # Adjust perturbation magnitude
                alpha=0.1       # Adjust SAR loss weight
            )
        else:
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

        final_score, total = evaluate_WVS_score.obtain_results("../data/WVQ_China_Evaluate.jsonl", pipe)

        print(f"WVS score: {final_score:.2f}%")
        print(f"Total valid responses: {total}")
        file.write(f"WVS score: {final_score:.2f}%\n")
        file.write(f"Total valid responses: {total}\n")
        file.write("\n")

print("Finished hyperparameter search. Results are saved in results_SFT_quant_reg_search.txt.")