import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import torch
import argparse
from tqdm import tqdm
import re

model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


with open("./data/WVQ_China_Train.jsonl", "r", encoding="utf-8") as file:
    first_line = file.readline().strip()  # Read only the first line
    data = json.loads(first_line)  # Convert JSON string to Python dictionary

# Print the result
print(data['messages'][0:2])
# print(data['messages'][2])

# messages = [
#     {"role": "system", "content": "You are an China chatbot that know China very well."},
#     {"role": "user", "content": "Do you agree with One of my main goals in life has been to make my parents proud? 1. Strongly agree 2. agree 3. Disagree 4. Strongly disagree. Give your single numerical answer with no explanation."},
# ]
outputs = pipe(
    data['messages'][0:2],
    max_new_tokens=256,
)

print("HERE'S THE OUTPUT")
print(outputs[0]["generated_text"][-1]['content'])