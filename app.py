import torch
import source.system as ss
from source.db import ChromaDB
from source.dataset import load_dataset
from source.utils import save_results
import argparse

def runLLM(model, reflection_prompt_template, path_results, sbert, episodic_mode=True, verbose=True, inference_mode=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm = ss.load_model(model, device)
    db = ChromaDB("./chroma_db", "episodic_memory", sbert)
    reflect = ss.create_reflection_pipeline(reflection_prompt_template, llm)
    
    if inference_mode:
        dataset = load_dataset(50)
        results = ss.run_inference(llm=llm, db=db, reflect=reflect, dataset=dataset)
        save_results(path_results, results)
    else:
        ss.run_chat(llm=llm, db=db, reflect=reflect, episodic_mode=episodic_mode, verbose=verbose)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run the LLM and begin your conversation")
    parser.add_argument("-m", "--model", help= "Model's Name", default="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
    parser.add_argument("-r", "--reflection_prompt", help= "Reflection Prompt Template", default="prompts/reflection_prompt_template.txt")
    parser.add_argument("--results", help= "Results output path", default="results/results.json")
    parser.add_argument("-e", "--episodic", help= "Run the LLM with the episodic memory module", action="store_true")
    parser.add_argument("-v", "--verbose", help= "Verbose", action= "store_true")
    parser.add_argument("-s", "--sbert", help= "SBERT Model to evaluate the similarity scores", default="all-mpnet-base-v2")
    parser.add_argument("-i", "--inference_mode", help= "Use inference mode instead of chat mode", action="store_true")
    namespace = parser.parse_args()
    
    runLLM(model=namespace.model,
           reflection_prompt_template=namespace.reflection_prompt, 
           path_results=namespace.results, 
           sbert=namespace.sbert, 
           episodic_mode=namespace.episodic,
           verbose=namespace.verbose, 
           inference_mode=namespace.inference_mode)
    