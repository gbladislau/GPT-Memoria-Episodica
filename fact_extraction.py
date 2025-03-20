import random
import json

# Function to extract facts
def extract_facts(dataset, num_examples = 10):
    facts = []
    
    answers = dataset.get("answer")[:num_examples] 
    contexts = dataset.get("search_results")[:num_examples]
    
    for i,c in enumerate(contexts):
        # correct_answers = answers[i].get("normalized_aliases")
        search_contexts = c.get("search_context")
        if (len(search_contexts) > 0):
            facts.append(search_contexts[0])
        else:
            print("NO CONTEXT")

    return facts


def loadJSON():
    with open("trivia_qa.json", "r") as f:
        dataset = json.load(f)
        
    facts = extract_facts(dataset, 100)
    return facts, dataset.get("question")
    
    
    
    
# def factsPrompt(llm, facts):
#     intro_message = "I will start giving you some facts and later ask questions about them\n"
    
#     llm(intro_message, max_new_tokens=100)[0]["generated_text"]
    
#     for i, fact in enumerate(facts):
#         message = f"FACT {i}\n" + fact + "END OF FACT {i}\n\n"
#         llm(message, max_new_tokens=100)[0]["generated_text"]


facts, questions = loadJSON()
for f in facts[:2]:
    print("Starting fact:\n")
    print(f)
    print("Ending fact\n")

print(questions[:2])

# factsPrompt(facts)