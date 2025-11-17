import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import os

from vocabulary import tokenize_de, tokenize_en

class TranslationDataset(Dataset):
    """
    A PyTorch Dataset for loading parallel text data.
    """
    def __init__(self, data_dir, de_vocab, en_vocab, mode='train'):
        self.de_path = os.path.join(data_dir, f'{mode}.de')
        self.en_path = os.path.join(data_dir, f'{mode}.en')
        
        with open(self.de_path, 'r', encoding='utf-8') as f:
            self.de_sents = f.read().strip().split('\n')
        with open(self.en_path, 'r', encoding='utf-8') as f:
            self.en_sents = f.read().strip().split('\n')
            
        self.de_vocab = de_vocab
        self.en_vocab = en_vocab

    def __len__(self):
        return len(self.de_sents)

    def __getitem__(self, idx):
        de_sent = self.de_sents[idx]
        en_sent = self.en_sents[idx]
        
        de_tokens = [self.de_vocab.stoi.get(t, self.de_vocab.stoi["<unk>"]) for t in tokenize_de(de_sent)]
        en_tokens = [self.en_vocab.stoi.get(t, self.en_vocab.stoi["<unk>"]) for t in tokenize_en(en_sent)]
        
        de_tensor = torch.tensor([self.de_vocab.stoi["<sos>"]] + de_tokens + [self.de_vocab.stoi["<eos>"]])
        en_tensor = torch.tensor([self.en_vocab.stoi["<sos>"]] + en_tokens + [self.en_vocab.stoi["<eos>"]])
        
        return en_tensor, de_tensor

def get_collate_fn(pad_idx):
    """
    Returns a collate_fn function that pads sequences in a batch.
    """
    def collate_fn(batch):
        en_batch, de_batch = zip(*batch)
        de_padded = pad_sequence(de_batch, batch_first=True, padding_value=pad_idx)
        en_padded = pad_sequence(en_batch, batch_first=True, padding_value=pad_idx)
        return en_padded, de_padded
    return collate_fn