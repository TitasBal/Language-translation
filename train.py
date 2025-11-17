import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import spacy
import os
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import Seq2SeqTransformer, generate_masks


spacy_en = spacy.load("en_core_web_sm")
spacy_de = spacy.load("de_core_news_sm")

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def tokenize_de(text):
    return [tok.text.lower() for tok in spacy_de.tokenizer(text)]

class Vocab:
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

class TranslationDataset(Dataset):
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
    def collate_fn(batch):
        en_batch, de_batch = zip(*batch)
        de_padded = pad_sequence(de_batch, batch_first=True, padding_value=pad_idx)
        en_padded = pad_sequence(en_batch, batch_first=True, padding_value=pad_idx)
        return en_padded, de_padded
    return collate_fn

if __name__ == "__main__":
    FREEZE_ENCODER_MEMORY = False
    mode = "frozen_encoder" if FREEZE_ENCODER_MEMORY else "baseline"
    print(f"--- RUNNING IN '{mode.upper()}' MODE (EN -> DE) ---")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    D_MODEL = 256
    NHEAD = 4
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    DIM_FEEDFORWARD = 1024
    
    data_dir = ".data"
    with open(os.path.join(data_dir, "train.de"), 'r', encoding='utf-8') as f:
        de_sents = f.read().strip().split('\n')
    with open(os.path.join(data_dir, "train.en"), 'r', encoding='utf-8') as f:
        en_sents = f.read().strip().split('\n')
        
    specials = ["<unk>", "<pad>", "<sos>", "<eos>"]
    de_vocab = Vocab(tokenize_de, de_sents, specials, min_freq=1)
    en_vocab = Vocab(tokenize_en, en_sents, specials, min_freq=1)
    PAD_IDX = en_vocab.stoi["<pad>"]
    
    print(f"Device: {DEVICE}")
    print(f"English (Source) Vocab Size: {len(en_vocab)}")
    print(f"German (Target) Vocab Size: {len(de_vocab)}")
    
    train_dataset = TranslationDataset(data_dir, de_vocab=de_vocab, en_vocab=en_vocab, mode='train')
    val_dataset = TranslationDataset(data_dir, de_vocab=de_vocab, en_vocab=en_vocab, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=get_collate_fn(PAD_IDX))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=get_collate_fn(PAD_IDX))
    
    model = Seq2SeqTransformer(
        src_vocab_size=len(en_vocab),
        tgt_vocab_size=len(de_vocab),
        d_model=D_MODEL, nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        freeze_encoder=FREEZE_ENCODER_MEMORY
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    train_losses, val_losses = [], []
    try:
        for epoch in range(EPOCHS):
            epoch_train_loss, epoch_val_loss = 0, 0
            total_steps = len(train_loader) + len(val_loader)
            with tqdm(total=total_steps, desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
                model.train()
                for src, tgt in train_loader:
                    src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                    tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
                    src_pad_mask, tgt_pad_mask, tgt_sub_mask = generate_masks(src, tgt_input, PAD_IDX, DEVICE)
                    
                    optimizer.zero_grad()
                    output = model(src, tgt_input, src_pad_mask, tgt_pad_mask, src_pad_mask, tgt_sub_mask)
                    loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item()
                    pbar.set_postfix(train_loss=f"{loss.item():.4f}")
                    pbar.update(1)

                model.eval()
                with torch.no_grad():
                    for src, tgt in val_loader:
                        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                        tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
                        src_pad_mask, tgt_pad_mask, tgt_sub_mask = generate_masks(src, tgt_input, PAD_IDX, DEVICE)
                        output = model(src, tgt_input, src_pad_mask, tgt_pad_mask, src_pad_mask, tgt_sub_mask)
                        loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
                        epoch_val_loss += loss.item()
                        pbar.set_postfix(val_loss=f"{loss.item():.4f}")
                        pbar.update(1)

            avg_epoch_train_loss = epoch_train_loss / len(train_loader)
            avg_epoch_val_loss = epoch_val_loss / len(val_loader)
            train_losses.append(avg_epoch_train_loss)
            val_losses.append(avg_epoch_val_loss)
            print(f"Epoch {epoch+1} Summary: Avg Train Loss: {avg_epoch_train_loss:.4f} | Avg Val Loss: {avg_epoch_val_loss:.4f}")
        
        with open(f"epoch_losses_{mode}.txt", "w") as f:
            for tl, vl in zip(train_losses, val_losses):
                f.write(f"{tl},{vl}\n")
    except KeyboardInterrupt:
        print("\n\nTraining interrupted. Saving artifacts...")

    print("\n--- Saving model and vocab ---")
    torch.save(model.state_dict(), f"model_{mode}.pt")
    torch.save(de_vocab, "vocab_de.pt")
    torch.save(en_vocab, "vocab_en.pt")

    if train_losses and val_losses:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title(f"Training and Validation Loss ({mode})")
        plt.xlabel("Epoch"), plt.ylabel("Loss"), plt.legend(), plt.grid(True)
        plt.savefig(f"loss_plot_{mode}.png")
        print(f"\nSaved loss plot to 'loss_plot_{mode}.png'")