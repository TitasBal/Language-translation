import torch
import sys
from tqdm import tqdm
from torchtext.data.metrics import bleu_score
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from model import Seq2SeqTransformer
from config import config
from vocabulary import load_tokenizer

def translate_sentence(model, sentence, tokenizer, device, max_len=50):
    model.eval()

    src_tokens = tokenizer.encode(sentence).ids
    src_tensor = torch.LongTensor([tokenizer.token_to_id("<sos>")] + src_tokens + [tokenizer.token_to_id("<eos>")]).unsqueeze(0).to(device)
    
    src_padding_mask = (src_tensor == tokenizer.token_to_id("<pad>")).to(device)

    tgt_tokens = [tokenizer.token_to_id("<sos>")]
    
    for _ in range(max_len):
        tgt_tensor = torch.LongTensor(tgt_tokens).unsqueeze(0).to(device)
        
        tgt_mask = model.transformer.generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)

        with torch.no_grad():
            output = model(src_tensor, tgt_tensor, src_padding_mask, None, src_padding_mask, tgt_mask)
        
        pred_token = output.argmax(2)[:, -1].item()
        tgt_tokens.append(pred_token)

        if pred_token == tokenizer.token_to_id("<eos>"):
            break
            
    decoded_sentence = tokenizer.decode(tgt_tokens, skip_special_tokens=True)
    return decoded_sentence

def evaluate_model(model, tokenizer, device):
    print("\n--- Evaluating Model Performance ---")
    
    test_en_path = f"{config['data_dir']}/val.en"
    test_de_path = f"{config['data_dir']}/val.de"
    
    with open(test_en_path, 'r', encoding='utf-8') as f:
        src_sentences = f.read().strip().split('\n')
    with open(test_de_path, 'r', encoding='utf-8') as f:
        ref_sentences = f.read().strip().split('\n')

    hypotheses = []
    references = []

    for src_sent, ref_sent in tqdm(zip(src_sentences, ref_sentences), total=len(src_sentences), desc="Translating Test Set"):
        hyp_sent = translate_sentence(model, src_sent, tokenizer, device)
        hypotheses.append(hyp_sent)
        references.append(ref_sent)
    
    hypotheses_tokenized = [tokenizer.encode(h).tokens for h in hypotheses]
    references_tokenized = [[tokenizer.encode(r).tokens] for r in references]

    bleu = bleu_score(hypotheses_tokenized, references_tokenized)
    print(f"\n--- BLEU Score on Test Set: {bleu*100:.2f} ---")
    
    return hypotheses, references

def show_examples(hypotheses, references, src_sentences, num_examples=5):
    print("\n--- Translation Examples ---")
    for i in range(num_examples):
        print(f"\n----------- Example {i+1} -----------")
        print(f"Source (EN):      {src_sentences[i]}")
        print(f"Reference (DE):   {references[i]}")
        print(f"Prediction (DE):  {hypotheses[i]}")
        print("---------------------------------")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluation.py <path_to_model.pt>")
        sys.exit(1)

    model_path = sys.argv[1]
    DEVICE = config['device']

    tokenizer = load_tokenizer(config['tokenizer_path'])
    tokenizer.decoder = ByteLevelDecoder()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Loaded tokenizer with a vocabulary size of {vocab_size}")

    model = Seq2SeqTransformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=config['d_model'], 
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    ).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"Model loaded from '{model_path}' and moved to {DEVICE}.")

    with open(f"{config['data_dir']}/val.en", 'r', encoding='utf-8') as f:
        src_sentences_for_examples = f.read().strip().split('\n')

    hypotheses, references = evaluate_model(model, tokenizer, DEVICE)
    show_examples(hypotheses, references, src_sentences_for_examples, num_examples=5)