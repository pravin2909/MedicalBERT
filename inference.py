# inference.py
import torch
import json
from model import MedicalBERT
from config import Config
from dataset import clean_text

def load_vocab():
    with open(Config.vocab_path) as f:
        token_to_id = json.load(f)

    vocab = {
        "stoi": token_to_id,
        "itos": {i: w for w, i in token_to_id.items()}
    }
    return vocab

def predict(text):
    # Load vocab
    vocab = load_vocab()

    # Load label map
    with open(Config.label_map_path) as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}

    num_classes = len(label2id)

    # Load model
    model = MedicalBERT(num_classes)
    model.load_state_dict(torch.load(Config.model_path, map_location="cpu"))
    model.eval()

    # Clean input
    text = clean_text(text)

    # Tokenize using your custom vocab
    words = text.split()

    ids = []
    ids.append(vocab["stoi"][Config.cls_token])

    for w in words:
        ids.append(vocab["stoi"].get(w, vocab["stoi"][Config.unk_token]))

    ids.append(vocab["stoi"][Config.sep_token])

    # Pad
    if len(ids) < Config.max_len:
        ids += [vocab["stoi"][Config.pad_token]] * (Config.max_len - len(ids))
    else:
        ids = ids[:Config.max_len]

    x = torch.tensor([ids], dtype=torch.long)

    # Predict
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()

    return id2label[pred]

if __name__ == "__main__":
    print(predict("High fever with chills and sweating in cycles"))
