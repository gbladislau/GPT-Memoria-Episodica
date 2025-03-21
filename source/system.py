import json
from transformers import pipeline
from utils import print_log
from tqdm import tqdm


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


def run_chat(llm, db, reflect, episodic_mode=False, user_inputs=None, verbose=False):
    print_log(verbose, "\nSTARTING CONVERSATION")
    print_log(verbose, f"EPISODIC MODE: {str(episodic_mode).upper()}")

    default_system_prompt = format_message(
        "You are a helpful AI Assistant. Answer the User's queries succinctly in one sentence.",
        is_user=False
    )

    memory = {
        "prev_convs": [],
        "what_worked": set(),
        "what_to_avoid": set(),
        "key_insights": set()
    }

    messages = [default_system_prompt]
    while True:
        user_input = (user_inputs and next(user_inputs)) or input("\nuser: ").strip()

        if user_input.lower() == "exit":
            conversation = format_conversation(messages)
            add_episodic_memory(reflect, conversation, db)
            print_log(verbose, "MEMORY SAVED")
            break

        if user_input.lower() == "exit_quiet":
            break

        user_message = format_message(user_input, is_user=True)
        messages.append(user_message)

        if episodic_mode:
            system_prompt = episodic_system_prompt(user_input, memory, db) or default_system_prompt
            messages = [system_prompt] + messages[1:]            

        response_data = llm(messages, max_new_tokens=300)[0]["generated_text"]
        assistant_response = response_data[-1]["content"]
        messages.append(format_message(assistant_response))

        print_log(verbose, "\nassistant:", assistant_response)

    print_log(verbose, "ENDING CONVERSATION")
    return messages


######################################################################


def create_reflection_pipeline(prompt_template_path, llm):
    prompt_template = ""
    with open(prompt_template_path, 'r') as file:
        prompt_template = file.read()

    return lambda conversation: json.loads(llm([{
        "role": "user",
        "content": prompt_template.format(conversation=conversation)
    }], max_new_tokens=300)[0]["generated_text"][-1]["content"])


def add_episodic_memory(reflect, conversation, db):
    reflection = reflect(conversation)
    db.insert({
        "conversation": conversation,
        **reflection
    })


def episodic_recall(query, db):
    return db.query(query)


def episodic_system_prompt(query, memory, db):
    data = episodic_recall(query, db)
    if not data: return None

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
    Previous Conversations: {' | '.join(memory['prev_convs'][:-1])}
    What has worked well: {'. '.join(memory['what_worked'])}
    What to avoid: {'. '.join(memory['what_to_avoid'])}
    Key insights: {'. '.join(memory['key_insights'])}
    
    Use these memories as context for your response to the user."""
    
    return format_message(episodic_prompt, is_user=False)


######################################################################


def run_inference(llm, db, reflect, dataset):
    contexts, questions, answers = list(zip(*dataset))

    result = {"non_episodic": [], "episodic": [], "answers": answers}

    for context in tqdm(contexts, desc="[ contexts ]"):
        messages = run_chat(llm, db, reflect, episodic_mode=True, user_inputs=iter((context, "exit")), verbose=False)

    for question in tqdm(questions, desc="[ questions ]"):
        for label, value in [("non_episodic", False), ("episodic", True)]:
            messages = run_chat(llm, db, reflect, episodic_mode=value, user_inputs=iter((question, "exit_quiet")), verbose=False)
            result[label].append(messages[-1]["content"])

    return result
