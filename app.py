import torch
import utils as ut

if __name__ == "__main__":
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    llm = ut.load_model("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4", device)
    reflect = ut.create_reflection_pipeline("reflection_prompt_template.txt", llm)

    print(ut.run_chat(llm))
