from transformers import pipeline
import torch
import json

def load_model(model_id, device):
    return pipeline("text-generation", model=model_id, device=device)

def format_message(query, is_user=False):
    return {
        "role": "user" if is_user else "assistant",
        "content": query
    }

def format_conversation(messages):
    conversation = []
    for message in messages[1:]:
        conversation.append(f"{message['role']}: {message['content']}")
    
    return "\n".join(conversation)

def run_chat(llm):
    system_prompt = format_message(
        "You are a helpful AI Assistant. Answer the User's queries succinctly in one sentence.",
        is_user=False
    )

    messages = [system_prompt]
    while True:

        user_message = format_message(input("\nuser: "), is_user=True)

        if user_message["content"].lower() == "exit": break
        else: messages.append(user_message)

        messages = llm(messages)[0]["generated_text"]
        response = messages[-1]["content"]
        print("\nassistant: ", response)

    return format_conversation(messages)

def create_reflection_pipeline(prompt_template_path, llm):
    prompt_template = ""
    with open(prompt_template_path, 'r') as file:
        prompt_template = file.read()

    return lambda conversation: json.loads(llm([{
        "role": "user",
        "content": prompt_template.format(conversation=conversation)
    }], max_new_tokens=100)[0]["generated_text"][-1]["content"])

def add_episodic_memory(reflect, conversation, db):
    reflection = reflect(conversation)
    data = {
        "conversation": conversation,
        **reflection
    }

    # TODO: salvar no db
    # {{
    #     "conversation": string,
    #     "context_tags": [string, string, string],
    #     "conversation_summary": string,
    #     "what_worked": string,
    #     "what_to_avoid": string
    #     "key_insights": string
    # }}

def episodic_recall(query, db):
    # TODO: recuperar do db a partir da query
    # {{
    #     "conversation": string,
    #     "context_tags": [string, string, string],
    #     "conversation_summary": string,
    #     "what_worked": string,
    #     "what_to_avoid": string
    #     "key_insights": string
    # }}
    pass

def episodic_system_prompt(query, memory, db):
    data = episodic_recall(query, memory, db)
    curr_conv = data["conversation"]

    if curr_conv not in memory["prev_convs"]:
        memory["what_worked"].update(data["what_worked"].split(". "))
        memory["what_to_avoid"].update(data["what_to_avoid"].split(". "))
        memory["key_insights"].update(data["key_insights"].split(". "))

    memory["prev_convs"] = [
        conv for conv in memory["prev_convs"] if conv != curr_conv
    ][-3:] + [curr_conv]

    episodic_prompt = f"""You are a helpful AI Assistant. Answer the user's questions to the best of your ability.
    You recall similar conversations with the user, here are the details:

    Current Conversation Match: {curr_conv}
    Previous Conversations: {' | '.join(memory['prev_convs'][:3])}
    What has worked well: {'. '.join(memory['what_worked'])}
    What to avoid: {'. '.join(memory['what_to_avoid'])}
    Key insights: {'. '.join(memory['key_insights'])}
    
    Use these memories as context for your response to the user."""
    
    return format_message(episodic_prompt, is_user=False)
