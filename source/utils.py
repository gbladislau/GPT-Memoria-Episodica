import json
from datasets import load_dataset
from evaluate import get_all_scores

def print_log(verbose, *args, **kargs):
    if verbose: print(*args, **kargs)

def save_results(path, results):
    with open(path, 'w') as f:
        f.write(json.dumps(results, indent=4))
        
def save_as_json(dataset_path, num_samples):
    dataset = load_dataset("narrativeqa", split=f"validation[:{num_samples}]")

    with open(dataset_path, "w") as f:
        json.dump(dataset.to_dict(), f, indent=4)

    print(f"Dataset salvo em {dataset_path}")
    
def load_all_dataset():
    dataset = load_dataset("narrativeqa")

    questions = dataset["validation"]["question"][:3]  # Select 100 samples
    answers = dataset["validation"]["answer"][:3]  # Corresponding ground truth answers

    # Example AI-generated responses (replace with your model's answers)
    ai_answers = [x["normalized_aliases"][-1] for x in answers]  

    get_all_scores(questions, ai_answers, answers)
    