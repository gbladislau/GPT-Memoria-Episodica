import json


def extract_facts(dataset, num_facts = 10):
    data = zip(
        dataset.get("search_results"),
        dataset.get("question"),
        dataset.get("answer")
    )

    return sorted([
        (context.get("search_context")[0], question, answer.get("normalized_aliases"))
        for context, question, answer in data if len(context.get("search_context")) > 0
    ], key=lambda x: len(x[0]))[:num_facts]
        

def getDataLoader(json_path, num_facts):
    with open(json_path, 'r') as f:
        dataset = json.load(f)
        
    return extract_facts(dataset, num_facts)
