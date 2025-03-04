import json
import argparse
from pathlib import Path

def format_prompts(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                json_obj = json.loads(line.strip())
                question = json_obj['messages'][0]['content'] + " " + json_obj['messages'][1]['content']
                answer = json_obj['messages'][2]['content']
                
                updated_obj = {"text": f"### Question: {question}\n ### Answer: {answer}"}

                outfile.write(json.dumps(updated_obj, ensure_ascii=False) + '\n')
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line as JSON. Copying without changes: {e}")
                outfile.write(line)

def main():
    parser = argparse.ArgumentParser(description='Format prompts in Llama style')
    parser.add_argument('--input_file', type=str, default='../data/WVQ_China_Train.jsonl', help='Path to the input JSONL file')
    parser.add_argument('--output_file', type=str, default="../data/WVQ_China_Train_Llama.jsonl", help='Path to the output JSONL file')
    
    args = parser.parse_args()
    
    # Ensure input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return 1
    
    # Replace text in the file
    try:
        format_prompts(args.input_file, args.output_file)
        print(f"Successfully processed '{args.input_file}'.")
        print(f"Output written to: {args.output_file}")
        return 0
    except Exception as e:
        print(f"Error: Failed to process file: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
