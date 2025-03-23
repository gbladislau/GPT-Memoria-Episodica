import torch
import source.system as ss
from source.db import ChromaDB
from source.dataset import load_dataset
from source.utils import save_results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm = ss.load_model("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4", device)
    db = ChromaDB("./chroma_db", "episodic_memory", "all-mpnet-base-v2")
    reflect = ss.create_reflection_pipeline("reflection_prompt_template.txt", llm)

    print(ss.run_chat(llm, db, reflect, episodic_mode=True, verbose=True))

    dataset = load_dataset(50)
    results = ss.run_inference(llm, db, reflect, dataset)

    save_results("results/new_results.json", results)
