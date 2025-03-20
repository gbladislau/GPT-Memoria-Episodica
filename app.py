import torch
import utils as ut
import chromadb
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    ## LLM
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    llm = ut.load_model("hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", device)
    ## DB
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="episodic_memory")
    # EMBEDDING MODEL
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    #REFLECTION PIPEL
    reflect = ut.create_reflection_pipeline("reflection_prompt_template.txt", llm)

    # Talk and then reflect
    conversation = ut.run_chat(llm)
    reflection = reflect(conversation)
    print(reflection)
    
    # Add reflection to the db
    ut.add_episodic_memory(reflect, conversation, collection, embedding_model)

    print(f"\n{reflection}\n\n{conversation}")
