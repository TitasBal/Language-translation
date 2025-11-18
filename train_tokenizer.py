import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

DATA_DIR = ".data"
VOCAB_SIZE = 30000
OUTPUT_DIR = "tokenizers"
os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = Tokenizer(BPE(unk_token="<unk>"))

tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE, 
    min_frequency=2,
    special_tokens=[
        "<pad>",
        "<sos>",
        "<eos>",
        "<unk>",
    ]
)

files = [
    os.path.join(DATA_DIR, "train.en"),
    os.path.join(DATA_DIR, "train.de")
]
print("Training BPE tokenizer...")
tokenizer.train(files, trainer)
print("Training complete.")

output_path = os.path.join(OUTPUT_DIR, "tokenizer.json")
tokenizer.save(output_path)
print(f"Tokenizer saved to '{output_path}'")