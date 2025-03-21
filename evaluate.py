from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

# def get_similarity_score(answer1, answer2, model):
#     """Compute cosine similarity between two answers"""
#     embeddings = model.encode([answer1, answer2])
#     return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

# def gen_avarage_similarity_score(questions, ai_answers, answers):
#     # Load the sentence embedding model
#     model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight & fast

#     scores = []
#     for i in range(len(ai_answers)):
#         similarity = get_similarity_score(ai_answers[i], answers[i]["normalized_value"], model)
#         scores.append(similarity)
#         print(f"Q: {questions[i]}")
#         print(f"AI Answer: {ai_answers[i]}")
#         print(f"Ground Truth: {answers[i]["normalized_value"]}")
#         print(f"Similarity Score: {similarity:.2f}\n")

#     # Average similarity score
#     average_score = sum(scores) / len(scores)
#     print(f"Average Similarity Score: {average_score:.2f}")
    
    
def getSimilarityScore(model, true_answer, ai_answers):
    embedding_true = model.encode(true_answer)
    embedding_ai = model.encode(ai_answers)
    
    similarities = model.similarity(embedding_true, embedding_ai)
    
    return torch.Tensor.numpy(similarities).reshape(-1, 1)



def score(model, true_answer, ai_answers):
    sim = getSimilarityScore(model, true_answer, ai_answers)
    return max(sim)

def getAllScores(data_answers, ai_all_answers):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    assert(len(data_answers) == len(ai_all_answers))
    num_samples = len(data_answers)
    
    scores = []
    
    for i in range(num_samples):
        s = score(model, data_answers[i], ai_all_answers[i])
        scores.append(s[0])
        
    return scores
    


true_answer = ["Flamengo", "Santos"]
ai_answers = [["Flamengos", "C.R.Flamengo", "Clube de Regatas do Flamengo"],
              ["Santos", "Santastico", "Pele FC"]]
    
scores = getAllScores(true_answer, ai_answers)

print(scores)


        
        
        
        