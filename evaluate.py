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

def extract_from_input(line):
    data = json.loads(line.strip())
    messages = data['messages']
    
    prompt, answer = None, None
    starting_choice, ending_choice = 0, 5
    
    for msg in messages:
        if msg['role'] == 'user':
            prompt = msg['content']
            starting_choice, ending_choice = extract_first_two_numbers(prompt)
        elif msg['role'] == 'assistant':
            answer = int(msg['content'])

    return prompt, answer, starting_choice, ending_choice

def format_prompt(prompt, starting_choice, ending_choice):
    return f"""You are a helpful Chinese chatbot that knows China very well. You only provide numerical answers.

Question: {prompt}

Provide your answer as a single number between {starting_choice} and {ending_choice}. 

Answer: """

def obtain_results(file_name, pipe):
    score, total = 0, 0
    
    # First count total lines
    with open(file_name, 'r') as f:
        total_lines = sum(1 for _ in f)
    
    # Then process the file with tqdm
    with open(file_name, 'r') as f:
        pbar = tqdm(f, total=total_lines, desc="Evaluating")
        for line in pbar:
            prompt, correct_answer, starting_choice, ending_choice = extract_from_input(line)
            
            # Format the prompt to encourage a direct numerical response
            formatted_prompt = format_prompt(prompt, starting_choice, ending_choice)

            response = pipe(formatted_prompt)[0]['generated_text']
            # Extract only the new text after our prompt
            new_text = response[len(formatted_prompt):]
            
            predicted_number = extract_number_from_response(new_text)
            
            if predicted_number is not None:
                total += 1
                len_range = ending_choice - starting_choice
                current_score = (len_range - abs(predicted_number - correct_answer)) / float(len_range)
                score += current_score
                
                # Update progress bar description with current score
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
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            pad_token_id=2,
            eos_token_id=2,
        )
    else:  # local adapter
        # Load the adapter config
        peft_config = PeftConfig.from_pretrained(model_path)
        
        # Load base model and tokenizer
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
            
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
        
        # Load the adapter
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        
        # Create pipeline with the adapted model
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            pad_token_id=2,
            eos_token_id=2,
        )

    return pipe

def main():
    parser = argparse.ArgumentParser(description='Evaluate a language model on QA pairs')
    parser.add_argument('--qa_file', type=str, help='Path to JSONL file containing QA pairs')
    parser.add_argument('--model_type', type=str, choices=['huggingface', 'local'],
                      help='Type of model to load (huggingface or local)')
    parser.add_argument('--model_path', type=str,
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