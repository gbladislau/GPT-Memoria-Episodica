from datasets import load_dataset
from evaluate import gen_avarage_similarity_score
import json

def saveAsJSON(dataset_path, num_samples):
    dataset = load_dataset("trivia_qa", "unfiltered", split=f"validation[:{num_samples}]")

    with open(dataset_path, "w") as f:
        json.dump(dataset.to_dict(), f, indent=4)

    print(f"Dataset salvo em {dataset_path}")
    
def loadALLDataset():
    dataset = load_dataset("trivia_qa", "unfiltered")

    questions = dataset["validation"]["question"][:3]  # Select 100 samples
    answers = dataset["validation"]["answer"][:3]  # Corresponding ground truth answers

    print(questions)
    print(type(questions))
    print(answers)

    # Example AI-generated responses (replace with your model's answers)
    ai_answers = [x["normalized_aliases"][-1] for x in answers]  

    gen_avarage_similarity_score(questions, ai_answers, answers)
    
    

saveAsJSON("trivia_qa.json", 100)


