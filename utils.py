from transformers import pipeline
import torch
import json

def load_model(model_id, device):
    return pipeline("text-generation", model=model_id, device=device)

def format_message(input_message, is_user=False):
    return {
        "role": "user" if is_user else "assistant",
        "content": input_message
    }

def format_conversation(messages):
    conversation = []
    for message in messages[1:]:
        conversation.append(f"{message['role']}: {message['content']}")
    
    return "\n".join(conversation)

def create_reflection_pipeline(prompt_template_path, llm):
    prompt_template = ""
    with open(prompt_template_path, 'r') as file:
        prompt_template = file.read()

    return lambda conversation: json.loads(llm([{
        "role": "user",
        "content": prompt_template.format(conversation=conversation)
    }], max_new_tokens=100)[0]["generated_text"][-1]["content"])

def run_chat(llm):
    system_prompt = format_message(
        "You are a helpful AI Assistant. Answer the User's queries succinctly in one sentence.",
        is_user=False
    )

    messages = [system_prompt]
    while True:

        user_message = format_message(input("\nuser: "), is_user=True)
        
        if user_message['content'].lower() == "exit": break
        else: messages.append(user_message)

        messages = llm(messages)[0]["generated_text"]
        response = messages[-1]["content"]
        print("\nassistant: ", response)

    return format_conversation(messages)
