from datasets import load_dataset
from evaluate import gen_avarage_similarity_score


dataset = load_dataset("trivia_qa", "unfiltered")

questions = dataset["validation"]["question"][:3]  # Select 100 samples
answers = dataset["validation"]["answer"][:3]  # Corresponding ground truth answers

# Example AI-generated responses (replace with your model's answers)
ai_answers = [x["normalized_aliases"][-1] for x in answers]  

gen_avarage_similarity_score(questions, ai_answers, answers)