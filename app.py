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
        results = ss.run_inference(llm, db, reflect, dataset)
        save_results(path_results, results)
    else:
        ss.run_chat(llm, db, reflect, episodic_mode, verbose)
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run the LLM and begin your conversation")
    parser.add_argument("-m", "--model", help= "Model's Name", default="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
    parser.add_argument("-r, --reflection_prompt", help= "Reflection Prompt Template", default="reflection_prompt_template.txt")
    parser.add_argument("-e", "--episodic", help= "Run the LLM with the episodic memory module", action="store_true")
    parser.add_argument("-v", "--verbose", help= "Verbose", action= "store_true")
    parser.add_argument("-s", "SBERT", help= "SBERT Model to evaluate the similarity scores", default="all-mpnet-base-v2")
    parser.add_argument("-i", "--inference_mode", help= "Use inference mode instead of chat mode", action="store_true")
    namespace = parser.parse_args()
    
    runLLM(namespace.model, namespace.reflection_prompt, 
           namespace.path_results, namespace.SBERT, 
           namespace.episodic_mode, namespace.verbose, 
           namespace.inference_mode)
    