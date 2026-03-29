import json
import torch
from torch.utils.data import Dataset
from config import Config
import re
import os

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

class Vocab:
    def __init__(self):
        self.stoi = {}
        self.itos = {}

    def build(self, texts):
        freq = {}
        for t in texts:
            for w in t.split():
                freq[w] = freq.get(w, 0) + 1

        words = sorted(freq, key=freq.get, reverse=True)
        words = words[: Config.vocab_size - 4]

        vocab = [
            Config.pad_token,
            Config.unk_token,
            Config.cls_token,
            Config.sep_token
        ] + words

        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.itos = {i: w for w, i in self.stoi.items()}

    def encode(self, text):
        ids = [self.stoi[Config.cls_token]]
        for w in text.split():
            ids.append(self.stoi.get(w, self.stoi[Config.unk_token]))
        ids.append(self.stoi[Config.sep_token])

        ids = ids[: Config.max_len]
        ids += [self.stoi[Config.pad_token]] * (Config.max_len - len(ids))
        return ids

class MedicalDataset(Dataset):
    def __init__(self, texts, labels, vocab, label2id):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.label2id = label2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(
                self.vocab.encode(self.texts[idx]), dtype=torch.long
            ),
            "labels": torch.tensor(
                self.label2id[self.labels[idx]], dtype=torch.long
            )
        }

def load_and_prepare_dataset():
    with open(Config.local_json_path) as f:
        data = json.load(f)

    texts = [clean_text(x["text"]) for x in data]
    labels = [x["label"] for x in data]

    unique_labels = sorted(set(labels))
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}

    split = int(0.8 * len(texts))
    train_texts, test_texts = texts[:split], texts[split:]
    train_labels, test_labels = labels[:split], labels[split:]

    vocab = Vocab()
    vocab.build(train_texts)

    os.makedirs("runs", exist_ok=True)
    with open(Config.vocab_path, "w") as f:
        json.dump(vocab.stoi, f, indent=2)
    with open(Config.label_map_path, "w") as f:
        json.dump(label2id, f, indent=2)

    train_ds = MedicalDataset(train_texts, train_labels, vocab, label2id)
    test_ds = MedicalDataset(test_texts, test_labels, vocab, label2id)

    print("Train:", len(train_ds), "Test:", len(test_ds))
    print("Classes:", len(label2id))

    return train_ds, test_ds, vocab, label2id, id2label
