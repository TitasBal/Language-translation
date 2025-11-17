import spacy
from collections import Counter

spacy_en = spacy.load("en_core_web_sm")
spacy_de = spacy.load("de_core_news_sm")

def tokenize_en(text):
    """
    Tokenizes English text into a list of lowercased tokens.
    """
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def tokenize_de(text):
    """
    Tokenizes German text into a list of lowercased tokens.
    """
    return [tok.text.lower() for tok in spacy_de.tokenizer(text)]

class Vocab:
    """
    A class to build a vocabulary from a list of sentences.
    """
    def __init__(self, tokenizer, sentences, specials, min_freq=1):
        self.tokenizer = tokenizer
        self.specials = specials
        self.itos = list(specials)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        
        counts = Counter(tok for sent in sentences for tok in self.tokenizer(sent))
        
        for tok, count in counts.items():
            if count >= min_freq:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def __len__(self):
        return len(self.itos)