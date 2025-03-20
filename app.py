import torch
import source.system as ss
from source.db import ChromaDB


if __name__ == "__main__":
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    llm = ss.load_model("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4", device)
    db = ChromaDB("./chroma_db", "episodic_memory", "all-MiniLM-L6-v2")
    reflect = ss.create_reflection_pipeline("reflection_prompt_template.txt", llm)

    conversation, messages = ss.run_chat(llm, db, reflect, True)

    print(f"\n{messages}\n\n{conversation}")
