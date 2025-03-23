# from datasets import load_dataset
# from evaluate import get_all_scores
import json

def saveAsJSON(dataset_path, num_samples):
    dataset = load_dataset("narrativeqa", split=f"validation[:{num_samples}]")

    with open(dataset_path, "w") as f:
        json.dump(dataset.to_dict(), f, indent=4)

    print(f"Dataset salvo em {dataset_path}")
    
def loadALLDataset():
    dataset = load_dataset("narrativeqa")

    questions = dataset["validation"]["question"][:3]  # Select 100 samples
    answers = dataset["validation"]["answer"][:3]  # Corresponding ground truth answers

    print(questions)
    print(type(questions))
    print(answers)

    # Example AI-generated responses (replace with your model's answers)
    ai_answers = [x["normalized_aliases"][-1] for x in answers]  

    get_all_scores(questions, ai_answers, answers)
    
def generate_evaluation_prompt(json_file, save_prompt_path):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    non_episodic = data["non_episodic"]
    episodic = data["episodic"]
    answers = data["answers"]
    
    prompt = (
        "Hello chat, you will be my evaluator. I want you to compare the answers from two models "
        "and say which one better resembled the correct answers. For each answer, I will give you "
        "a list of possible correct answers, the answer from model 1, "
        "and the answer from model 2. You must say your decision and explain it briefly\n\n"
    )
    
    for i, (expected, non_epi, epi) in enumerate(zip(answers, non_episodic, episodic), start=1):
        expected_answers = "\n".join(f"- {ans}" for ans in expected)
        prompt += (
            f"Answer {i}:\n"
            f"Expected possible answers:\n{expected_answers}\n"
            f"Answers from model 1: {non_epi}\n"
            f"Answers from model 2: {epi}\n\n"
        )
    
    with open(save_prompt_path, "w", encoding="utf-8") as file:
        file.write(prompt)
    
    

generate_evaluation_prompt("results.json", "evaluation_prompt.txt")


