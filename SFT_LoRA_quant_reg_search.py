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

class SARTrainer(SFTTrainer):
    def __init__(self, *args, epsilon=1e-3, alpha=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon  # Perturbation magnitude
        self.alpha = alpha      # SAR loss weight

    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass for standard loss
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        task_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

        # Generate adversarial perturbation
        inputs_embeds = model.get_input_embeddings()(inputs["input_ids"])
        inputs_embeds.requires_grad = True

        perturbed_outputs = model(inputs_embeds=inputs_embeds, attention_mask=inputs["attention_mask"])
        perturbed_logits = perturbed_outputs.logits

        loss_adv = F.cross_entropy(perturbed_logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        loss_adv.backward()

        perturbation = self.epsilon * inputs_embeds.grad.sign()
        perturbed_inputs = inputs_embeds + perturbation.detach()

        # Forward pass on perturbed inputs
        perturbed_outputs = model(inputs_embeds=perturbed_inputs, attention_mask=inputs["attention_mask"])
        perturbed_logits = perturbed_outputs.logits

        # Compute KL divergence loss
        sar_loss = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(perturbed_logits, dim=-1), reduction="batchmean")

        # Combine losses
        total_loss = task_loss + self.alpha * sar_loss

        return (total_loss, outputs) if return_outputs else total_loss

base_model = 'meta-llama/Llama-3.2-1B-Instruct'
dataset = load_dataset("json", data_files="./data/WVQ_China_Train.jsonl", split="train")

results = []

# LoRA, quantization, SAR
options = [[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]]

for option in options:
    print("Option (LoRA, quantization, SAR):", option)

    max_memory = {i: '46000MB' for i in range(torch.cuda.device_count())}
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        max_memory=max_memory
    )
    model.config.use_cache = False

    if options[1]: # use quantization
        compute_dtype = getattr(torch, "float16")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )

        model.quantization_config = quant_config

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

    if options[2]: # Use SAR
        trainer = SARTrainer(
            model=model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            args=training_params,
            epsilon=1e-3,   # Adjust perturbation magnitude
            alpha=0.1       # Adjust SAR loss weight
        )
    else:
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            args=training_params
        )

    if options[0]: # Use LoRA
        peft_params = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )

        trainer.peft_config = peft_params

    print("Fine-tuning model...")
    trainer.train()

    print("Finished fine-tuning. Beginning evaluation...")

    model = trainer.model
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()

    tokenizer = trainer.tokenizer

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    final_score, total = evaluate_WVS_score.obtain_results("./data/WVQ_China_Evaluate.jsonl", pipe)

    results.append((option, final_score, total))

with open("results_SFT_LoRA_quant_reg_search.txt", "w") as file:
    for result in results:
        option, final_score, total = result
        file.write(f"Setting (LoRA, quantization, SAR): {option}\n")
        file.write(f"WVS score: {final_score:.2f}%\n")
        file.write(f"Total valid responses: {total}\n")
        file.write("\n")

print("Finished hyperparameter search. Results are saved in results_SFT_LoRA_quant_reg_search.txt.")