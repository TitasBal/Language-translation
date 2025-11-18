from tokenizers import Tokenizer

def load_tokenizer(path):
    """
    Loads a trained tokenizer from its saved file.
    """
    tokenizer = Tokenizer.from_file(path)
    return tokenizer