import json

# Function to extract facts
def extract_facts(dataset, num_facts = 10):
    facts = []
    questions = dataset.get("question")[:num_facts]
    answers = dataset.get("answer")[:num_facts] 
    contexts = dataset.get("search_results")[:num_facts]
    
   
    for i in range(num_facts):
        question = questions[i]
        c_answers = answers[i].get("normalized_aliases")
        search_contexts = contexts[i].get("search_context")
        
        if (len(search_contexts) > 0):
            context = search_contexts[0]
            facts.append((question, c_answers, context))
        
    
    return facts
        

def getDataLoader(json_path, num_facts):
    with open(json_path, "r") as f:
        dataset = json.load(f)
        
    facts = extract_facts(dataset, num_facts)
    
    return facts


data = getDataLoader("trivia_qa.json", 100)

print(len(data))

