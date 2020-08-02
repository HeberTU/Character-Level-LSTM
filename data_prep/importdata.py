import os

def import_corpus(path: str):


    with open(path, 'r') as f:
        text = f.read()

    return text
