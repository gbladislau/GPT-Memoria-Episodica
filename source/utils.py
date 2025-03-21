import json

def print_log(verbose, *args, **kargs):
    if verbose: print(*args, **kargs)

def save_results(path, results):
    with open(path, 'w') as f:
        f.write(json.dumps(results, indent=4))
