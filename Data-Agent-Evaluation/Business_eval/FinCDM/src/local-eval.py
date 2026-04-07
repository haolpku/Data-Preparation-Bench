import json
import re
import torch
from transformers import AutoModel, AutoTokenizer

# Model configuration
base_model = 'path/to/your/model'  # Replace with your model path
device_map = 'cuda:0'  # Change to your preferred device
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModel.from_pretrained(
    base_model,
    trust_remote_code=True,
    device_map=device_map,
    torch_dtype=torch.bfloat16
)
model = model.eval()

# Regex filter class
class RegexFilter:
    def __init__(self, regex_pattern: str = r"(A|B|C|D)", group_select=0, fallback: str = "[invalid]"):
        self.regex = re.compile(regex_pattern)
        self.group_select = group_select
        self.fallback = fallback

    def apply(self, resp):
        match = self.regex.findall(resp)
        if match:
            match = match[self.group_select]
            if isinstance(match, tuple):
                match = [m for m in match if m]
                match = match[0] if match else self.fallback
            return match.strip()
        return self.fallback

# Load JSON file
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Inference and evaluation
def evaluate(json_file_path, num_trials=10, model_name="model"):
    data = load_json(json_file_path)
    filter = RegexFilter()
    results = []

    nums = 0

    for item in data:
        print('Current question:', nums)
        nums += 1
        query = item['query']
        gold_answer = item['answer']
        item_results = []

        # Run multiple trials
        for trial in range(num_trials):
            # Prepare input
            inputs = tokenizer(query, return_tensors='pt').to(device_map)
            
            # Model inference with temperature 1.0
            with torch.no_grad():
                pred = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=True,  # Enable sampling to use temperature
                    temperature=1.0,
                    repetition_penalty=1.0
                )
            
            # Decode output
            raw_output = tokenizer.decode(pred[0], skip_special_tokens=True)
            raw_output = raw_output.replace(query, "")
            # Extract answer
            predicted_answer = filter.apply(raw_output)
            
            # Save single trial result
            item_results.append({
                'trial': trial + 1,
                'raw_output': raw_output,
                'predicted_answer': predicted_answer,
                'gold_answer': gold_answer,
                'correct': 1 if predicted_answer == gold_answer else 0
            })
        
        # Save all test results for this question
        results.append({
            'id': item['id'],
            'query': query,
            'trials': item_results
        })
    
    # Calculate overall accuracy (based on average accuracy of all trials)
    total_trials = sum(len(item['trials']) for item in results)
    correct_count = sum(1 for item in results for trial in item['trials'] if trial['correct'])
    overall_accuracy = correct_count / total_trials if total_trials else 0
    
    # Calculate accuracy for each trial round
    trial_accuracies = []
    for trial_idx in range(num_trials):
        trial_correct = sum(1 for item in results if item['trials'][trial_idx]['correct'])
        trial_total = len(results)
        trial_accuracy = trial_correct / trial_total if trial_total else 0
        trial_accuracies.append({
            'trial': trial_idx + 1,
            'accuracy': trial_accuracy
        })
    
    # Save results to file
    output_filename = f'{model_name}_evaluation_results.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump({
            'results': results,
            'overall_accuracy': overall_accuracy,
            'trial_accuracies': trial_accuracies
        }, f, ensure_ascii=False, indent=2)
    
    return results, overall_accuracy

# Execute evaluation
if __name__ == "__main__":
    json_file_path = 'path/to/your/test.json'  # Replace with your JSON file path
    model_name = "your_model_name"  # Replace with your model name for output file
    results, accuracy = evaluate(json_file_path, num_trials=10, model_name=model_name)
    print(f"Evaluation completed, overall accuracy: {accuracy:.2%}")
    for item in results:
        print(f"\nQuestion ID: {item['id']}")
        for trial in item['trials']:
            print(f"  Trial {trial['trial']} - Predicted: {trial['predicted_answer']}, "
                  f"Gold answer: {trial['gold_answer']}, Correct: {trial['correct']}, "
                  f"Raw output: {trial['raw_output']}")