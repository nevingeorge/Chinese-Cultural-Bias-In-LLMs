import argparse
import evaluate
import random

def obtain_results(file_name):
    score, total = 0, 0
    
    with open(file_name, 'r') as f:
        for line in f:
            _, correct_answer, starting_choice, ending_choice = evaluate.extract_from_input(line)
            predicted_number = random.randint(starting_choice, ending_choice)
            
            total += 1
            len_range = ending_choice - starting_choice
            current_score = (len_range - abs(predicted_number - correct_answer)) / float(len_range)
            score += current_score

    final_score = (score / total * 100) if total > 0 else 0
    return final_score, total

def main():
    parser = argparse.ArgumentParser(description='Evaluate a language model on QA pairs')
    parser.add_argument('--qa_file', type=str, help='Path to JSONL file containing QA pairs')
    args = parser.parse_args()

    print("Beginning evaluation...")

    running_score = 0
    for i in range(1000):
        final_score, _ = obtain_results(args.qa_file)
        running_score += final_score
    avg_score = running_score / 1000

    print("\nEvaluation complete!")
    print(f"Final Results:")
    print(f"Score: {avg_score:.2f}%")

if __name__ == "__main__":
    main()