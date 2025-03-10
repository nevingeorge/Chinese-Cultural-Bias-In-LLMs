import random
import pandas as pd
import json
import os

# File paths (Update these paths as needed)
PROMPT_FILE = "text-infilling.xlsx"
DATASET_WRITE_FILE_PATH = "data/dpo/dpo_dataset.json"
CATEGORIES = ['Author', 'Beverage', 'Clothing', 'Food', 'Location', 'Name', 'Religious', 'Sports']


# Load prompts with [MASK] from an Excel file based on category
def load_prompts_by_category(prompt_file, category_type):
    df = pd.read_excel(prompt_file)
    filtered_df = df[df["Entity Type"] == category_type].copy()
    filtered_df["Prompt"] = filtered_df["Prompt"].str.replace(r"\[MASK\]\.$", "", regex=True)
    return filtered_df["Prompt"].tolist()

# Load entity lists from an Excel file
def load_entities(entity_file):
    df = pd.read_excel(entity_file, header=None)
    return df.iloc[:, 0].dropna().tolist()  # Read first column, drop NaN values

# Generate multiple positive and negative pairs for each prompt
def generate_dpo_pairs(prompts, chinese_entities, western_entities, num_samples=10):
    dpo_data = []
    
    for prompt in prompts:
        # Randomly sample multiple positive and negative entities for each prompt
        positive_samples = random.sample(chinese_entities, min(num_samples, len(chinese_entities)))
        negative_samples = random.sample(western_entities, min(num_samples, len(western_entities)))
        
        for positive_entity, negative_entity in zip(positive_samples, negative_samples):
            # positive_example = f"{positive_entity}."
            # negative_example = f"{negative_entity}."

            # dpo_data.append([prompt, positive_example, negative_example])
            dpo_data.append([prompt, positive_entity, negative_entity])

    return dpo_data

if __name__ == "__main__":
    
    dpo_pairs = []
    for category in CATEGORIES:

        prompts_file = f"data/finetune/prompts/{PROMPT_FILE}"
        chinese_entities_file = f"data/finetune/entities/zh/{category.lower()}_zh.xlsx"
        western_entities_file = f"data/finetune/entities/en/{category.lower()}_en.xlsx"
        # Load data
        prompts = load_prompts_by_category(prompts_file, category)  # Change category if needed
        chinese_entities = load_entities(chinese_entities_file)
        western_entities = load_entities(western_entities_file)

        # Generate DPO training pairs
        dpo_pairs += generate_dpo_pairs(prompts, chinese_entities, western_entities)

    if not os.path.exists("data/dpo"):
        os.makedirs("data/dpo")
    # Save dataset to JSON for fine-tuning
    with open(DATASET_WRITE_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(dpo_pairs, f, ensure_ascii=False, indent=4)

    print(f"DPO dataset generated successfully and saved as {DATASET_WRITE_FILE_PATH}!")