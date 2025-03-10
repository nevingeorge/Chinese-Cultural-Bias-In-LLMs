import os

from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    TrainingArguments,
)
import trl
from peft import LoraConfig

# MODEL_NAME = "sft-dpo-LoRA_64_128_0"

# # Model Configuration
# SFT_ADAPTER_SAVE_PATH = "finetuned_models/sft-LoRA_64_128_0"
# DPO_ADAPTER_SAVE_PATH = f"finetuned_models/{MODEL_NAME}"

# HUGGING_FACE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
# EXISTING_SFT_ADAPTER_PATH = ""
# CASUAL_LM = LlamaForCausalLM

# # Data Configuration
# SFT_DATASET_PATH = "data/dpo/entity_sentences.json"
# DPO_DATASET_PATH = "data/dpo/dpo_dataset.json"


# # CBS Configuration
# PROMPT_PATH = "data/test/prompts/text-infilling.xlsx"
# CATEGORIES = ['Author', 'Beverage', 'Clothing', 'Food', 'Location', 'Name', 'Religious', 'Sports']

class Args:
    def __init__(self):
        # Model Configuration
        self.model_name = ""
        self.sft_adapter_save_path = ""
        self.dpo_adapter_save_path = ""
        self.huggingface_model_name = "meta-llama/Llama-3.2-1B-Instruct"
        self.existing_sft_adapter_path = ""

        self.casual_lm = LlamaForCausalLM  # Ensure imported
        self.dpo_trainer = trl.DPOTrainer
        self.dpo_config = trl.DPOConfig

        # Data Configuration
        self.sft_dataset_path = "data/dpo/entity_sentences.json"
        self.dpo_dataset_path = "data/dpo/dpo_dataset.json"

        # CBS Configuration
        self.prompt_path = "data/test/prompts/text-infilling.xlsx"
        self.categories = ['Author', 'Beverage', 'Clothing', 'Food', 'Location', 'Name', 'Religious', 'Sports']

        # LoRA Configuration
        self.lora_r = 64
        self.lora_alpha = 128
        self.lora_dropout = 0

        # Training Configuratiuon
        self.sft_n_epochs = 3
        self.sft_lr = 5e-05
        self.dpo_n_epochs = 3
        self.dpo_lr = 1e-06



class LoRAConfig:
    def __init__(self, r, alpha, dropout):
        assert isinstance(r, int), "LORA_R must be an integer!"
        assert isinstance(alpha, int), "LORA_ALPHA must be an integer!"
        assert isinstance(dropout, (int, float)), "LORA_DROPOUT must be a number (int or float)!"
        
        self.r = r
        self.alpha = alpha
        self.dropout = dropout

class TrainConfig:
    def __init__(self, sft_n_epochs=3, sft_lr=5e-5,
                        dpo_n_epochs=3, dpo_lr=1e-6):
        self.sft_n_epochs = sft_n_epochs
        self.sft_lr = sft_lr

        self.dpo_n_epochs = dpo_n_epochs
        self.dpo_lr = dpo_lr

def create_lora_config(loraConfig: LoRAConfig):
    return LoraConfig(
        r=loraConfig.r,
        lora_alpha=loraConfig.alpha,
        lora_dropout=loraConfig.dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

def create_sft_training_args(output_dir, trainArgs: TrainConfig):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=trainArgs.sft_n_epochs,
        learning_rate=trainArgs.sft_lr,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_dir=f"{output_dir}/logs",
        logging_steps=25,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )

def create_dpo_training_args(output_dir, TypeConfig, trainArgs: TrainConfig):
    assert(TypeConfig == trl.DPOConfig or TypeConfig == trl.ORPOConfig)
    return TypeConfig(
        output_dir=output_dir,
        num_train_epochs=trainArgs.dpo_n_epochs,
        learning_rate=trainArgs.dpo_lr,
        logging_dir=f"{output_dir}/logs",
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )

class SFTConfig:
    def __init__(self, args: Args):
        # parameters for model
        self.sft_adapter_save_path = args.sft_adapter_save_path
        self.huggingface_model_name = args.huggingface_model_name
        self.existing_sft_adapter_path = args.existing_sft_adapter_path
        self.casual_lm = args.casual_lm
        # self.peft_config = args.peft_config
        # self.training_args = args.training_args
        lora_config = LoRAConfig(args.lora_r, args.lora_alpha, args.lora_dropout)
        self.peft_config = create_lora_config(lora_config)
        train_config = TrainConfig(sft_n_epochs=args.sft_n_epochs,
                                   sft_lr = args.sft_lr,
                                   dpo_n_epochs=args.dpo_n_epochs,
                                   dpo_lr=args.dpo_lr)
        self.training_args = create_sft_training_args(self.sft_adapter_save_path, train_config)

        # parameters for data
        assert args.sft_dataset_path.endswith('.json'), "sft_dataset_path must be a JSON file"
        self.sft_dataset_path = args.sft_dataset_path

class DPOConfig:
    def __init__(self, args: Args):
        # parameters for models
        self.model_name = args.model_name
        self.dpo_adapter_save_path = args.dpo_adapter_save_path
        self.huggingface_model_name = args.huggingface_model_name
        # self.sft_model_adapter_path = args.sft_model_adapter_path
        self.sft_model_adapter_path = args.existing_sft_adapter_path if args.existing_sft_adapter_path != "" else args.sft_adapter_save_path
        self.casual_lm = args.casual_lm
        # self.peft_config = peft_config
        # self.training_args = training_args
        lora_config = LoRAConfig(args.lora_r, args.lora_alpha, args.lora_dropout)
        self.peft_config = create_lora_config(lora_config)
        train_config = TrainConfig(sft_n_epochs=args.sft_n_epochs,
                                   sft_lr = args.sft_lr,
                                   dpo_n_epochs=args.dpo_n_epochs,
                                   dpo_lr=args.dpo_lr)
        self.training_args = create_dpo_training_args(self.dpo_adapter_save_path, args.dpo_config, train_config)

        self.config = args.dpo_config
        self.trainer = args.dpo_trainer

        # parameters for data
        assert args.dpo_dataset_path.endswith('.json'), "dpo_dataset_path must be a JSON file"
        self.dpo_dataset_path = args.dpo_dataset_path

class CBSConfig:
    def __init__(self, args: Args):
        self.model_name = args.model_name

        self.huggingface_model_name = args.huggingface_model_name
        self.dpo_adapter_save_path = args.dpo_adapter_save_path
        self.casual_lm = args.casual_lm

        self.prompt_path = args.prompt_path
        self.categories = args.categories

# class Config:
#     VALID_CATEGORIES = ('Author', 'Beverage', 'Clothing', 'Food',
#                         'Location', 'Name', 'Religious', 'Sports')
#     # VALID_HUGGING_FACE_MODEL_NAMES = {"meta-llama/Llama-3.2-1B-Instruct"}

#     def __init__(self, category_type, culture_type, model_name, prompt_path):
#         if category_type not in self.VALID_CATEGORIES:
#             raise ValueError(f"Invalid category type: {category_type}. Must be one of {self.VALID_CATEGORIES}")
#         # if hugging_face_model_name not in self.VALID_HUGGING_FACE_MODEL_NAMES:
#         #     raise ValueError(f"Invalid Huggging Face model_name: {model_name}. Must be one of {self.VALID_HUGGING_FACE_MODEL_NAMES}")
        
#         self.category_type = category_type
#         self.culture_type = culture_type
#         self.model_name = model_name
#         self.prompt_path = prompt_path

#         self.cat_cul = f"{self.category_type.lower()}_{self.culture_type.lower()}"
#         self.entity_path = f"data/test/entities/{self.culture_type}/{self.cat_cul}.xlsx"

#         probs_folder_path = f"CBS/{model_name}/probs/"
#         if not os.path.exists(probs_folder_path):
#             os.makedirs(probs_folder_path)
#         self.prob_path = f"CBS/{model_name}/probs/prob_{self.cat_cul}.xlsx"