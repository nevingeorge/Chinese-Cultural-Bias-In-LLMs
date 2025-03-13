import argparse
from pathlib import Path

def split_jsonl(input_file, output_prefix):
    """
    Split a JSONL file into two equal parts.
    
    Args:
        input_file (str): Path to the input JSONL file
        output_prefix (str): Prefix for output files (will be appended with _part1.jsonl and _part2.jsonl)
    """
    # Read all lines from the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Calculate the midpoint
    mid_point = len(lines) // 2
    
    # Create the first output file with the first half
    output_file1 = f"{output_prefix}_part1.jsonl"
    with open(output_file1, 'w', encoding='utf-8') as f:
        f.writelines(lines[:mid_point])
    
    # Create the second output file with the second half
    output_file2 = f"{output_prefix}_part2.jsonl"
    with open(output_file2, 'w', encoding='utf-8') as f:
        f.writelines(lines[mid_point:])
    
    return output_file1, output_file2

def main():
    parser = argparse.ArgumentParser(description='Split a JSONL file into two equal parts')
    parser.add_argument('--input_file', type=str, default='./WVQ_China_1000_modified.jsonl', help='Path to the input JSONL file')
    parser.add_argument('--output-prefix', type=str, default='split', 
                        help='Prefix for output files (default: "split")')
    
    args = parser.parse_args()
    
    # Ensure input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return 1
    
    # Split the file
    try:
        part1, part2 = split_jsonl(args.input_file, args.output_prefix)
        print(f"Successfully split '{args.input_file}':")
        print(f"  First half written to: {part1}")
        print(f"  Second half written to: {part2}")
        return 0
    except Exception as e:
        print(f"Error: Failed to split file: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
