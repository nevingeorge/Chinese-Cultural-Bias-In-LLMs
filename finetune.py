import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType

# Load the pre-trained Llama 3.2 model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Load the CultureBank dataset from Hugging Face Hub
dataset = load_dataset("SALT-NLP/CultureBank")
dataset_cc = concatenate_datasets([dataset["tiktok"], dataset["reddit"]])

# Filter dataset to only include rows where "cultural group" == "Chinese"
filtered_dataset = dataset_cc.filter(lambda x: x["cultural group"] == "Chinese")

# Keep only "eval_whole_desc" column
def preprocess_function(examples):
    return {"text": examples["eval_whole_desc"]}

# Filtered_dataset has 292 rows
filtered_dataset = filtered_dataset.map(preprocess_function, remove_columns=filtered_dataset.column_names)

def tokenize_function(examples):
    encoding = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding

# Apply tokenization
train_dataset = filtered_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05
)

# Wrap the model with LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Check trainable params

# Custom tqdm progress bar for training
class TQDMProgressBar(TrainerCallback):
    def __init__(self):
        self.progress_bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.progress_bar = tqdm(total=state.max_steps, desc="Training Progress", unit="step")

    def on_step_end(self, args, state, control, **kwargs):
        self.progress_bar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        self.progress_bar.close()

training_args = TrainingArguments(
    output_dir="./results",
    save_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=1, 
    num_train_epochs=8,
    weight_decay=0.05,
    optim="paged_adamw_32bit",
    bf16=True,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[TQDMProgressBar()],
)

trainer.train()

output_dir = "./fine-tuned-llama-3.2-lora"
trainer.save_model(output_dir)

output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)

print("Fine-tuning complete! Model saved to './fine-tuned-llama-3.2-lora'.")