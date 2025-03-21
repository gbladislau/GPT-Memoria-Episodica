from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import evaluate as ev
from source.dataset import load_dataset
import json
    
def _get_similarity_score(model, true_answer, ai_answers):
    embedding_true = model.encode(true_answer)
    embedding_ai = model.encode(ai_answers)
    
    similarities = model.similarity(embedding_true, embedding_ai)
    
    return torch.Tensor.numpy(similarities).reshape(-1, 1)

def _score(model, true_answer, ai_answers):
    sim = _get_similarity_score(model, true_answer, ai_answers)
    return max(sim)

def get_all_scores(data_answers, ai_all_answers):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    assert(len(data_answers) == len(ai_all_answers))
    num_samples = len(data_answers)
    
    scores = []
    
    scores = np.zeros(num_samples)
    for i in range(num_samples):
        s = _score(model, data_answers[i], ai_all_answers[i])
        scores[i] = s[0]
        
    return scores

def gen_evaluation(results_path='./results.json'):
    
    with open(results_path) as f:
        results_dict = json.load(f)
   
    no_memory_scores = ev.get_all_scores(results_dict["answers"], results_dict["non_episodic"])
    memory_scores = ev.get_all_scores(results_dict["answers"], results_dict["episodic"])
    
    
    print(f"No memory mean: {no_memory_scores.mean()} std: {no_memory_scores.std()}")
    print(f"Memory mean: {memory_scores.mean()}, std: {memory_scores.std()}")
    print(f"Percentage of improvment: {(memory_scores.mean()/no_memory_scores.mean())*100}")
        

if __name__ == "__main__":
    # print("Small test")
    # true_answer = ["Flamengo", "Santos"]
    # ai_answers = [["Flamengos", "C.R.Flamengo", "Clube de Regatas do Flamengo"],
    #             ["Santos", "Santastico", "Pele FC"]]
        
    # scores = get_all_scores(true_answer, ai_answers)

    # print(scores)
    gen_evaluation()



        
        