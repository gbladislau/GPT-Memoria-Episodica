import datasets

def load_dataset(n=None):
    ds = datasets.load_dataset("deepmind/narrativeqa")

    contexts = list(map(lambda document: document["summary"]["text"], ds["validation"]["document"]))
    questions = list(map(lambda question: question["text"], ds["validation"]["question"]))
    answers = list(map(lambda answers: list(map(lambda answer: answer["text"], answers)), ds["validation"]["answers"]))

    return sorted(zip(contexts, questions, answers), key=lambda x: len(x[0]))[slice(n)]
