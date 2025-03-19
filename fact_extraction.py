import random

# Function to extract facts
def extract_facts(dataset, num_facts=10):
    facts = []
    
    for item in dataset["train"]:
        question = item["question"]
        answers = item["answer"]["value"]  # List of possible correct answers
        contexts = item["search_results"]["search_context"]  # Context passages

        if not contexts:  # Skip if no context available
            continue

        # Search for a sentence containing the answer
        for context in contexts:
            sentences = context.split(". ")  # Split into sentences
            for sentence in sentences:
                if any(answer.lower() in sentence.lower() for answer in answers):
                    facts.append(sentence.strip())
                    break  # Take only the first relevant sentence
        
        if len(facts) >= num_facts:
            break

    return facts


