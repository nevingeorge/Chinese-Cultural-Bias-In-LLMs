import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
import torch
import argparse
from tqdm import tqdm
import re

def extract_number_from_response(response):
    # Extract the first number from the model's response
    numbers = re.findall(r'\d+', response)
    if len(numbers) == 0:
        return None
    return int(numbers[0])
        
def extract_first_two_numbers(prompt):
    numbers = re.findall(r'\d+', prompt)
    return int(numbers[0]), int(numbers[1])

def obtain_results(file_name, pipe):
    score, total = 0, 0
    
    with open(file_name, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    with open(file_name, 'r') as f:
        pbar = tqdm(f, total=total_lines, desc="Evaluating")
        for line in pbar:
            data = json.loads(line.strip()) 

            correct_answer = int(data['messages'][2]['content'])
            starting_choice, ending_choice = extract_first_two_numbers(data['messages'][1]['content'])

            outputs = pipe(
                data['messages'][0:2],
                max_new_tokens=10,
                pad_token_id=128001,
            )
            response = outputs[0]["generated_text"][-1]['content']
            predicted_number = extract_number_from_response(response)
            
            if predicted_number is not None:
                total += 1
                len_range = ending_choice - starting_choice
                current_score = (len_range - abs(predicted_number - correct_answer)) / float(len_range)
                score += current_score
                
                current_avg_score = (score / total * 100)
                pbar.set_postfix({'Score': f'{current_avg_score:.2f}%'})

    final_score = (score / total * 100) if total > 0 else 0
    return final_score, total

def load_model(model_type, model_path):
    """Load either a Hugging Face model or a local PEFT adapter model."""

    if model_type == "huggingface":
        pipe = pipeline(
            "text-generation", 
            model=model_path, 
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
    else:  # local adapter
        peft_config = PeftConfig.from_pretrained(model_path)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
            
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        model = PeftModel.from_pretrained(base_model, model_path, is_trainable=False)
        model = model.merge_and_unload()
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    return pipe

def main():
    parser = argparse.ArgumentParser(description='Evaluate a language model on QA pairs')
    parser.add_argument('--qa_file', type=str, default='./data/WVQ_China_Evaluate.jsonl', help='Path to JSONL file containing QA pairs')
    parser.add_argument('--model_type', type=str, default='huggingface', choices=['huggingface', 'local'],
                      help='Type of model to load (huggingface or local)')
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-3.2-1B-Instruct',
                      help='Path to model (Hugging Face model name or local adapter path)')
    
    args = parser.parse_args()

    print(f"Loading {args.model_type} model from {args.model_path}...")
    pipe = load_model(args.model_type, args.model_path)
    print("Model loaded successfully!")

    print("Beginning evaluation...")
    final_score, total = obtain_results(args.qa_file, pipe)
    
    print("\nEvaluation complete!")
    print(f"Final Results:")
    print(f"Score: {final_score:.2f}%")
    print(f"Total valid responses: {total}")

if __name__ == "__main__":
    main()