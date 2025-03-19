from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def get_similarity_score(answer1, answer2, model):
    """Compute cosine similarity between two answers"""
    embeddings = model.encode([answer1, answer2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def gen_avarage_similarity_score(questions, ai_answers, answers):
    # Load the sentence embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight & fast

    scores = []
    for i in range(len(ai_answers)):
        similarity = get_similarity_score(ai_answers[i], answers[i]["normalized_value"], model)
        scores.append(similarity)
        print(f"Q: {questions[i]}")
        print(f"AI Answer: {ai_answers[i]}")
        print(f"Ground Truth: {answers[i]["normalized_value"]}")
        print(f"Similarity Score: {similarity:.2f}\n")

    # Average similarity score
    average_score = sum(scores) / len(scores)
    print(f"Average Similarity Score: {average_score:.2f}")