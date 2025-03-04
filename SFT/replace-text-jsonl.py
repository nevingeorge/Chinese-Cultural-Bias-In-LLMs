import json
import argparse
from pathlib import Path

def replace_text_in_jsonl(input_file, output_file, old_text, new_text):
    """
    Replace all instances of old_text with new_text in a JSONL file.
    
    Args:
        input_file (str): Path to the input JSONL file
        output_file (str): Path to the output JSONL file
        old_text (str): Text to be replaced
        new_text (str): Replacement text
    """
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Parse the JSON object
            try:
                json_obj = json.loads(line.strip())
                
                # Replace text in all string values recursively
                def replace_in_obj(obj):
                    if isinstance(obj, str):
                        return obj.replace(old_text, new_text)
                    elif isinstance(obj, list):
                        return [replace_in_obj(item) for item in obj]
                    elif isinstance(obj, dict):
                        return {k: replace_in_obj(v) for k, v in obj.items()}
                    else:
                        return obj
                
                updated_obj = replace_in_obj(json_obj)
                
                # Write the updated JSON object
                outfile.write(json.dumps(updated_obj, ensure_ascii=False) + '\n')
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line as JSON. Copying without changes: {e}")
                outfile.write(line)

def main():
    parser = argparse.ArgumentParser(description='Replace text in a JSONL file')
    parser.add_argument('--input_file', type=str, default="../data/WVQ_China_1000.jsonl", help='Path to the input JSONL file')
    parser.add_argument('--output_file', type=str, default="../data/WVQ_China_1000_modified.jsonl", help='Path to the output JSONL file')
    parser.add_argument('--old-text', type=str, 
                        default="You can only choose one option.",
                        help='Text to be replaced (default: "You can only choose one option.")')
    parser.add_argument('--new-text', type=str,
                        default="Give your single numerical answer with no explanation.",
                        help='Replacement text (default: "Give your single numerical answer with no explanation.")')
    
    args = parser.parse_args()
    
    # Ensure input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return 1
    
    # Replace text in the file
    try:
        replace_text_in_jsonl(args.input_file, args.output_file, args.old_text, args.new_text)
        print(f"Successfully processed '{args.input_file}'.")
        print(f"Output written to: {args.output_file}")
        return 0
    except Exception as e:
        print(f"Error: Failed to process file: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
