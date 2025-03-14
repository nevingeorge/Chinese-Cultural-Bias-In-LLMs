import torch
import pandas as pd
import numpy as np
import os

from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

from config import Args, DPOConfig, CBSConfig
import utils

class Config:
    VALID_CATEGORIES = ('Author', 'Beverage', 'Clothing', 'Food',
                        'Location', 'Name', 'Religious', 'Sports')

    def __init__(self, category_type, culture_type, model_name, prompt_path):
        if category_type not in self.VALID_CATEGORIES:
            raise ValueError(f"Invalid category type: {category_type}. Must be one of {self.VALID_CATEGORIES}")
        
        self.category_type = category_type
        self.culture_type = culture_type
        self.model_name = model_name
        self.prompt_path = prompt_path

        self.cat_cul = f"{self.category_type.lower()}_{self.culture_type.lower()}"
        self.entity_path = f"data/test/entities/{self.culture_type}/{self.cat_cul}.xlsx"

        probs_folder_path = f"CBS/{model_name}/probs/"
        if not os.path.exists(probs_folder_path):
            os.makedirs(probs_folder_path)
        self.prob_path = f"CBS/{model_name}/probs/prob_{self.cat_cul}.xlsx"

def get_prompts_by_category(file_path, category_type):
    df = pd.read_excel(file_path)
    filtered_df = df[df["Entity Type"] == category_type].copy()
    filtered_df["Prompt"] = filtered_df["Prompt"].str.replace(r"\[MASK\]\.$", "", regex=True)
    return filtered_df["Prompt"].tolist()

def get_entities(file_path):
    df = pd.read_excel(file_path)
    return df['Entity'].tolist()

def get_prob_for_entities(config: Config, model, tokenizer, device):
    prompts = get_prompts_by_category(config.prompt_path, config.category_type)
    entities = get_entities(config.entity_path)
    entity_probs = {}

    for i, entity in enumerate(entities):
        entity_ids = tokenizer(entity, add_special_tokens=False)["input_ids"]
        entity_log_prob = np.zeros(len(prompts))

        for idx, token_id in enumerate(entity_ids):
            if idx == 0:
                inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
            else:
                prefix = entity_ids[:idx]
                modified_prompts = [prompt + " " + tokenizer.decode(prefix) for prompt in prompts]
                
                # Tokenize the modified texts with padding
                inputs = tokenizer(modified_prompts, padding=True, return_tensors="pt").to(device)
            
            inputs = {k: (v.to(dtype=model.dtype) if v.dtype.is_floating_point else v.to(torch.long)) 
                        for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
            
            logits = outputs.logits

            last_nonpad_indices = inputs["attention_mask"].sum(dim=1) - 1  # (batch_size,)
            last_logits = logits[torch.arange(logits.shape[0]).to(device), last_nonpad_indices, :]  # (batch_size, vocab_size)

            prob_distribution = torch.nn.functional.softmax(last_logits, dim=-1)

            prob = torch.log(prob_distribution.to(torch.float32)[:, int(token_id)]).cpu().numpy()

            entity_log_prob += prob  # Add log probabilities for each token

        entity_probs[entity] = entity_log_prob


    df = pd.DataFrame.from_dict(entity_probs, orient="index", columns=prompts)
    df = df.reset_index().rename(columns={"index": "Entity"})
    df.to_excel(config.prob_path, index=False)

def compute_cbs_score(zh_prob_path, en_prob_path, output_path):
    df_zh = pd.read_excel(zh_prob_path)
    df_en = pd.read_excel(en_prob_path)

    assert list(df_zh.columns) == list(df_en.columns), "Mismatch in prompt columns!"

    total_pairs = len(df_en) * len(df_zh)

    prompts = df_zh.columns[1:]
    
    scores = {}

    for prompt in prompts:
        count = 0
        for en_prob in df_en[prompt]:
            for zh_prob in df_zh[prompt]:
                if zh_prob >= en_prob:
                    count += 1

        scores[prompt] = count / total_pairs
    
    df_scores = pd.DataFrame(scores.items(), columns=["Prompt", "Score"])
    df_scores.to_excel(output_path, index=False)

    return scores

def get_average_cbs_score(categories, model_name, output_path):
    average_scores = []

    for category in categories:
        file_path = f"CBS/{model_name}/cbs/cbs_{category.lower()}.xlsx"
        try:
            df = pd.read_excel(file_path)

            if "Score" in df.columns:
                avg_score = df["Score"].mean()  # Compute average
                average_scores.append({"Category": category, "CBS": avg_score})
            else:
                print(f"Warning: 'Score' column not found in {file_path}, skipping.")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    df_avg = pd.DataFrame(average_scores)
    df_avg.to_excel(output_path, index=False)

def cbs(cbs_config):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model, tokenizer = utils.load_model_and_tokenizer(cbs_config.huggingface_model_name,
                                                      adapter_path=cbs_config.adapter_save_path,
                                                      casual_lm=cbs_config.casual_lm)
    model.eval()

    cbs_folder_path = f'CBS/{cbs_config.model_name}/cbs/'
    if not os.path.exists(cbs_folder_path):
        os.makedirs(cbs_folder_path)

    for category in cbs_config.categories:
        try:
            config_zh = Config(category_type=category,
                               culture_type='zh', 
                               model_name=cbs_config.model_name,
                               prompt_path=cbs_config.prompt_path)
            get_prob_for_entities(config_zh, model, tokenizer, device)
        except Exception as e:
            print(f"Error processing category '{category}', culture 'zh': {e}")

        try:
            config_en = Config(category_type=category,
                               culture_type='en', 
                               model_name=cbs_config.model_name,
                               prompt_path=cbs_config.prompt_path)
            get_prob_for_entities(config_en, model, tokenizer, device)
        except Exception as e:
            print(f"Error processing category '{category}', culture 'en': {e}")
        
        output_path = os.path.join(cbs_folder_path, f'cbs_{category.lower()}.xlsx')
        try:
            compute_cbs_score(config_zh.prob_path, config_en.prob_path, output_path)
        except Exception as e:
            print(f"Error copmuting CBS score category '{category}': {e}")

    
    output_path = os.path.join(cbs_folder_path, 'average_cbs.xlsx')
    get_average_cbs_score(cbs_config.categories, cbs_config.model_name, output_path)
    print(f"Done with cbs: {output_path}")

def main():
    args = Args()
    cbs_config = CBSConfig(args)
    cbs(cbs_config)

if __name__ == '__main__':
    main()