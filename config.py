import torch

class Config:
    # Dataset
    local_json_path = "bert_medical_5k.json"

    # Vocab
    vocab_size = 8000
    max_len = 64

    pad_token = "[PAD]"
    unk_token = "[UNK]"
    cls_token = "[CLS]"
    sep_token = "[SEP]"

    # Model
    embedding_dim = 256
    n_heads = 8
    n_layers = 4
    ff_dim = 512  # Reduced from 1024 to fit 4GB VRAM better
    dropout = 0.1

    # Training (optimized for GTX 1650 - 4GB VRAM)
    batch_size = 8  # Reduced from 32 to fit in VRAM
    gradient_accumulation_steps = 4  # Effective batch size = 8 * 4 = 32
    epochs = 15
    lr = 1e-4  # Learning rate works well with gradient accumulation
    seed = 42
    
    # Mixed precision training (FP16) - saves ~50% VRAM
    use_amp = True  # Automatic Mixed Precision

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    model_path = "runs/model.pt"
    vocab_path = "runs/vocab.json"
    label_map_path = "runs/label_map.json"
