import os
import json
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from dataset import load_and_prepare_dataset
from model import MedicalBERT
from config import Config


CHECKPOINT_PATH = "runs/checkpoint.pt"


def save_checkpoint(epoch, model, optimizer, best_val_acc):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_acc": best_val_acc
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"[Checkpoint] Saved at epoch {epoch}")


def load_checkpoint(model, optimizer):
    if not os.path.exists(CHECKPOINT_PATH):
        print("No checkpoint found. Starting fresh.")
        return 0, 0.0

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=Config.device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_acc = checkpoint["best_val_acc"]

    print(f"[Checkpoint] Loaded. Resuming from epoch {start_epoch}")
    return start_epoch, best_val_acc


def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(Config.device)
            labels = batch["labels"].to(Config.device)

            # Use mixed precision for evaluation too (faster)
            if Config.use_amp and Config.device == "cuda":
                with autocast():
                    logits = model(input_ids)
                    loss = loss_fn(logits, labels)
            else:
                logits = model(input_ids)
                loss = loss_fn(logits, labels)
            
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(1, len(dataloader))
    acc = correct / max(1, total)
    return avg_loss, acc


def train_model():
    print(f"Training on device: {Config.device}")
    torch.manual_seed(Config.seed)

    os.makedirs("runs", exist_ok=True)

    # Load data
    train_ds, test_ds, vocab, label2id, id2label = load_and_prepare_dataset()

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        test_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = MedicalBERT(num_classes=len(label2id)).to(Config.device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.lr,
        weight_decay=1e-2
    )

    start_epoch, best_val_acc = load_checkpoint(model, optimizer)
    loss_fn = nn.CrossEntropyLoss()
    
    # Mixed precision training scaler
    scaler = GradScaler() if Config.use_amp and Config.device == "cuda" else None
    if Config.use_amp and Config.device == "cuda":
        print("[AMP] Mixed precision training enabled (FP16)")

    # Training loop
    for epoch in range(start_epoch, Config.epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()  # Initialize gradients

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{Config.epochs}",
            leave=False
        )

        for step, batch in enumerate(progress):
            input_ids = batch["input_ids"].to(Config.device)
            labels = batch["labels"].to(Config.device)

            # Forward pass with mixed precision
            if Config.use_amp and scaler is not None:
                with autocast():
                    logits = model(input_ids)
                    loss = loss_fn(logits, labels)
                    # Scale loss for gradient accumulation
                    loss = loss / Config.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
            else:
                logits = model(input_ids)
                loss = loss_fn(logits, labels)
                loss = loss / Config.gradient_accumulation_steps
                loss.backward()

            running_loss += loss.item() * Config.gradient_accumulation_steps

            # Gradient accumulation: update weights every N steps
            if (step + 1) % Config.gradient_accumulation_steps == 0:
                if Config.use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()

            progress.set_postfix({
                "loss": running_loss / (step + 1),
                "eff_batch": Config.batch_size * Config.gradient_accumulation_steps
            })
        
        # Handle remaining gradients if last batch doesn't align with accumulation steps
        if len(train_loader) % Config.gradient_accumulation_steps != 0:
            if Config.use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        train_loss = running_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader)

        print(
            f"Epoch {epoch + 1}/{Config.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Config.model_path)

            with open(Config.label_map_path, "w") as f:
                json.dump(label2id, f, indent=2)

            print(f"[Model Saved] Best val_acc = {best_val_acc:.4f}")

        save_checkpoint(epoch, model, optimizer, best_val_acc)

    print("Training completed.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    train_model()
