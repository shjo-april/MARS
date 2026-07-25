import json

def read_json(filepath, encoding=None):
    with open(filepath, 'r', encoding=encoding) as f:
        data = json.load(f)
    return data

def write_json(filepath, data, encoding=None):
    with open(filepath, 'w', encoding=encoding) as f:
        json.dump(data, f, indent = '\t', ensure_ascii=False)

