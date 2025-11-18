import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import os

class TranslationDataset(Dataset):
    def __init__(self, data_dir, tokenizer, mode='train'):
        self.de_path = os.path.join(data_dir, f'{mode}.de')
        self.en_path = os.path.join(data_dir, f'{mode}.en')
        
        with open(self.de_path, 'r', encoding='utf-8') as f:
            self.de_sents = f.read().strip().split('\n')
        with open(self.en_path, 'r', encoding='utf-8') as f:
            self.en_sents = f.read().strip().split('\n')
            
        self.tokenizer = tokenizer
        self.sos_id = tokenizer.token_to_id("<sos>")
        self.eos_id = tokenizer.token_to_id("<eos>")

    def __len__(self):
        return len(self.de_sents)

    def __getitem__(self, idx):
        de_sent = self.de_sents[idx]
        en_sent = self.en_sents[idx]
        
        de_tokens = self.tokenizer.encode(de_sent).ids
        en_tokens = self.tokenizer.encode(en_sent).ids
        
        de_tensor = torch.tensor([self.sos_id] + de_tokens + [self.eos_id])
        en_tensor = torch.tensor([self.sos_id] + en_tokens + [self.eos_id])
        
        return en_tensor, de_tensor

def get_collate_fn(pad_idx):
    def collate_fn(batch):
        en_batch, de_batch = zip(*batch)
        de_padded = pad_sequence(de_batch, batch_first=True, padding_value=pad_idx)
        en_padded = pad_sequence(en_batch, batch_first=True, padding_value=pad_idx)
        return en_padded, de_padded
    return collate_fn