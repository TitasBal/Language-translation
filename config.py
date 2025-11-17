import torch

config = {
    "mode": "baseline",  # "baseline" or "frozen_encoder"
    "freeze_encoder": False, # If True, freezes the encoder and source embedding weights
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    
    "data_dir": ".data",
    "specials": ["<unk>", "<pad>", "<sos>", "<eos>"], # Special tokens
    "min_freq": 1, # Minimum frequency for a token to be included in the vocabulary

    "d_model": 256,
    "nhead": 4,
    "num_encoder_layers": 3,
    "num_decoder_layers": 3,
    "dim_feedforward": 1024,
    "dropout": 0.1,

    "batch_size": 64,
    "learning_rate": 1e-4,
    "epochs": 100,

    "model_save_path": "model_{mode}.pt",
    "loss_plot_path": "loss_plot_{mode}.png",
    "loss_log_path": "epoch_losses_{mode}.txt",
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