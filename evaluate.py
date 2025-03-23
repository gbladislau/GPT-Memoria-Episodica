from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import evaluate as ev
from source.dataset import load_dataset
import argparse
import json
    
def _get_similarity_score(model, true_answer, ai_answers):
    embedding_true = model.encode(true_answer)
    embedding_ai = model.encode(ai_answers)
    
    similarities = model.similarity(embedding_true, embedding_ai)
    
    return torch.Tensor.numpy(similarities).reshape(-1, 1)

def _score(model, true_answer, ai_answers):
    sim = _get_similarity_score(model, true_answer, ai_answers)
    return sim.mean()

def get_all_scores(data_answers, ai_all_answers):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    assert(len(data_answers) == len(ai_all_answers))
    num_samples = len(data_answers)
    
    scores = []
    
    scores = np.zeros(num_samples)
    for i in range(num_samples):
        s = _score(model, data_answers[i], ai_all_answers[i])
        scores[i] = s
        
    return scores

def gen_quantitative_evaluation(results_path='results/results.json', rerun=True, plot=True):
    
    with open(results_path) as f:
        results_dict = json.load(f)
    
    if rerun:
        no_memory_scores = ev.get_all_scores(results_dict["answers"], results_dict["non_episodic"])
        memory_scores = ev.get_all_scores(results_dict["answers"], results_dict["episodic"])    
        np.save("mem.npy", memory_scores)
        np.save("no_mem.npy", no_memory_scores)
    else:
        no_memory_scores = np.load("no_mem.npy")
        memory_scores = np.load("mem.npy")
        
    print(f"No memory mean: {no_memory_scores.mean()} std: {no_memory_scores.std()}")
    print(f"Memory mean: {memory_scores.mean()}, std: {memory_scores.std()}")
    print(f"Percentage of improvment: {(memory_scores.mean()/no_memory_scores.mean())*100}")

    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid") 

        plt.figure(figsize=(8, 6))
        sns.boxplot([no_memory_scores, memory_scores], patch_artist=True)
        plt.xticks([0,1], ["Memoryless", "Memory"])
        plt.tight_layout(pad=2)
        plt.ylim([0,1])
        plt.grid()
        plt.ylabel("Similarity Score")
        plt.title("Comparison of Scores with and without Memory")
        plt.savefig("boxplot.png")  # Save the figure
        plt.show()


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
    prompt+= """Give me an analysis based on 5 criterias: 
                1. **Accuracy & Performance** - How well does each model's answer match the ground truth? Is the information presented correctly and fully?
                2. **Consistency** - Are the answers provided by each model consistent when given similar inputs or prompts?
                3. **Contextual Relevance** - Does the model's answer remain relevant to the context of the question or task? Does the memory-enabled model show better contextual understanding due to its ability to remember past information?
                4. **Logical Coherence** - Is the reasoning behind the model's answer logically structured and sound? How well does each model handle complex reasoning tasks?
                5. **Generalization** - How well do the models handle out-of-distribution data or novel questions that were not part of the training set?
    

            """
    with open(save_prompt_path, "w", encoding="utf-8") as file:
        file.write(prompt)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run the evaluation module to calculate the LLM's results")
    parser.add_argument("--plot", "-p", help="Generate plot for quantitative"
                        "analysis", action="store_true")
    parser.add_argument("--dont_rerun", help="Use the mem.npy and no_mem.npy"
                        "scores previously calculated",
                        action="store_true")
    parser.add_argument("--gen_prompt", help="Generate qualitative prompt",
                        action="store_true")
    parser.add_argument("--result", help="LLM anwsers results input file path (json)", default="results/results.json")
    parser.add_argument("--prompt", help="Prompt file output path", default="prompts/evaluation_prompt.txt")
    namespace = parser.parse_args()
    
    gen_quantitative_evaluation(plot=namespace.plot,
                                rerun = not namespace.dont_rerun,
                                results_path= namespace.result)
    if namespace.gen_prompt:
        generate_evaluation_prompt(namespace.result,
                                   namespace.prompt)


        
        