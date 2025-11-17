import torch
import os
import re

run_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and re.match(r'^\d+_loss$', d)]
if not run_dirs:
    run_number = 1
else:
    existing_nums = [int(re.match(r'^(\d+)_loss$', d).group(1)) for d in run_dirs]
    run_number = max(existing_nums) + 1

weights_dir = "./weights"
run_dir = f"./{run_number}_loss"

os.makedirs(weights_dir, exist_ok=True)
os.makedirs(run_dir, exist_ok=True)

config = {
    "run_number": run_number,
    "mode": "baseline",
    "freeze_encoder": False,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    "data_dir": ".data",
    "specials": ["<unk>", "<pad>", "<sos>", "<eos>"],
    "min_freq": 1,

    "d_model": 256,
    "nhead": 4,
    "num_encoder_layers": 3,
    "num_decoder_layers": 3,
    "dim_feedforward": 1024,
    "dropout": 0.1,

    "batch_size": 64,
    "learning_rate": 1e-4,
    "epochs": 100,

    "model_save_path": os.path.join(weights_dir, f"model_{{mode}}_{run_number}.pt"),
    "loss_plot_path": os.path.join(run_dir, "loss_plot_{mode}.png"),
    "loss_log_path": os.path.join(run_dir, "epoch_losses_{mode}.txt"),
    "vocab_de_path": "vocab_de.pt",
    "vocab_en_path": "vocab_en.pt"
}

if config["mode"] == "frozen_encoder":
    config["freeze_encoder"] = True
    print("--- CONFIG: Running in FROZEN ENCODER mode ---")
else:
    config["freeze_encoder"] = False
    config["mode"] = "baseline"
    print("--- CONFIG: Running in BASELINE mode ---")

print(f"--- CONFIG: Saving run artifacts to '{run_dir}' and '{weights_dir}' ---")