from transformers import pipeline
import torch
import json
from sentence_transformers import SentenceTransformer

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
        user_input = input("\nuser: ").strip()
        print(f"{user_input}")
        if user_input.lower() == "exit":
            break

        user_message = format_message(user_input, is_user=True)
        messages.append(user_message)

        # Generate response
        response_data = llm(messages, max_new_tokens=100)[0]["generated_text"]
        # Ensure response is properly formatted
        assistant_response = response_data[-1]["content"]

        print("\nassistant:", assistant_response)

        # Append response to conversation
        messages.append(format_message(assistant_response))

    print("ENDING CONVERSATION")
    return format_conversation(messages)


def create_reflection_pipeline(prompt_template_path, llm):
    prompt_template = ""
    with open(prompt_template_path, 'r') as file:
        prompt_template = file.read()

    return lambda conversation: json.loads(llm([{
        "role": "user",
        "content": prompt_template.format(conversation=conversation)
    }], max_new_tokens=100)[0]["generated_text"][-1]["content"])


def add_episodic_memory(reflect, conversation, db, embedding_model):
    reflection = reflect(conversation)
    
    data = {
        "conversation": conversation,
        **reflection
    }

    # Generate embedding from conversation summary (or full conversation)
    vector = embedding_model.encode(data["conversation_summary"])
    
    # Store in ChromaDB
    db.add(
        ids=[conversation[:10]],  # Unique ID (first 10 chars as an example)
        embeddings=[vector],  # Store vector representation
        metadatas=[data]  # Store metadata
    )
    
    print("Memory added successfully!")


def episodic_recall(query, db, embedding_model, top_k=3):
    """Retrieves the most relevant episodic memories based on a query."""
    
    # Generate query embedding
    query_vector = embedding_model.encode(query).tolist()

    # Perform similarity search
    results = db.query(
        query_embeddings=[query_vector],
        n_results=top_k  # Retrieve top K relevant memories
    )

    # Format and return results
    retrieved_memories = []
    for i in range(len(results["ids"][0])):
        retrieved_memories.append({
            "conversation": results["metadatas"][0][i]["conversation"],
            "context_tags": results["metadatas"][0][i]["context_tags"],
            "conversation_summary": results["metadatas"][0][i]["conversation_summary"],
            "what_worked": results["metadatas"][0][i]["what_worked"],
            "what_to_avoid": results["metadatas"][0][i]["what_to_avoid"],
            "key_insights": results["metadatas"][0][i]["key_insights"],
        })

    return retrieved_memories


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
