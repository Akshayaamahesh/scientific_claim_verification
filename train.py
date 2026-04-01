"""
Fine-tune the NLI verifier on SciFact training data.

Usage:
    python train.py
    python train.py --epochs 5 --batch_size 4 --lr 1e-5
"""

import argparse
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
import config

LABEL2ID = {"SUPPORT": 0, "CONTRADICT": 1, "NEI": 2}
ID2LABEL = {0: "SUPPORT", 1: "CONTRADICT", 2: "NEI"}


class SciFastDataset(Dataset):
  
    def __init__(self, df, tokenizer, max_length=256):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = []

        for _, row in df.iterrows():
            claim    = str(row["claim"])
            gold_ev  = str(row["gold_evidence_text"]).strip()
            abstract = str(row["abstract"])
            label    = LABEL2ID[row["label"]]

            if gold_ev != "" and gold_ev != "nan":
                evidence = gold_ev
            else:
                sentences = [s.strip() for s in abstract.split(".") if s.strip()]
                evidence  = ". ".join(sentences[:2])

            if not evidence or evidence == "nan":
                evidence = abstract[:300]

            self.samples.append((claim, evidence, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        claim, evidence, label = self.samples[idx]

        encoding = self.tokenizer(
            claim,
            evidence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label":          torch.tensor(label, dtype=torch.long)
        }


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro",
                        zero_division=0)
    return acc, macro_f1


def train(epochs=5, batch_size=4, lr=1e-5, max_length=256):
    device = torch.device("cpu")
    print("Using device: cpu (MPS disabled for DeBERTa training stability)")

    print("\nLoading datasets...")
    train_df = pd.read_pickle(config.TRAIN_PATH)
    dev_df   = pd.read_pickle(config.DEV_PATH)

    print(f"Train size : {len(train_df)}")
    print(f"Dev size   : {len(dev_df)}")
    print(f"Train label distribution:\n{train_df['label'].value_counts()}\n")

    model_name = "cross-encoder/nli-deberta-v3-small"
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    train_dataset = SciFastDataset(train_df, tokenizer, max_length)
    dev_dataset   = SciFastDataset(dev_df,   tokenizer, max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01, eps=1e-6)

    accumulation_steps = 4

    total_steps  = (len(train_loader) // accumulation_steps) * epochs
    warmup_steps = total_steps // 10

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    print(f"\nStarting training: {epochs} epochs, "
          f"batch_size={batch_size}, lr={lr}")
    print(f"Gradient accumulation every {accumulation_steps} steps "
          f"(effective batch size = {batch_size * accumulation_steps})\n")

    best_f1         = 0.0
    best_model_path = "models/nli-finetuned"
    os.makedirs(best_model_path, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss            = 0
        all_preds, all_labels = [], []

        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = loss_fn(outputs.logits, labels) / accumulation_steps
            loss.backward()

            total_loss += loss.item() * accumulation_steps
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                effective_step = (step + 1) // accumulation_steps
                if effective_step % 10 == 0:
                    print(f"  Epoch {epoch+1} | "
                          f"Effective step {effective_step} | "
                          f"Loss: {loss.item() * accumulation_steps:.4f}")

        train_acc = accuracy_score(all_labels, all_preds)
        train_f1  = f1_score(all_labels, all_preds,
                             average="macro", zero_division=0)
        avg_loss  = total_loss / len(train_loader)

        dev_acc, dev_f1 = evaluate_model(model, dev_loader, device)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss : {avg_loss:.4f}")
        print(f"  Train Acc  : {train_acc:.3f} | Train F1 : {train_f1:.3f}")
        print(f"  Dev Acc    : {dev_acc:.3f}   | Dev F1   : {dev_f1:.3f}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"  ✓ Best model saved (Dev F1: {dev_f1:.3f})")

    print(f"\nTraining complete. Best Dev F1: {best_f1:.3f}")
    print(f"Model saved to: {best_model_path}")
    print("\nNext step — update config.py:")
    print('  NLI_MODEL = "models/nli-finetuned"')

    return best_model_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Fine-tune NLI verifier on SciFact training data"
    )
    parser.add_argument("--epochs",     type=int,   default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--batch_size", type=int,   default=4,
                        help="Batch size per step (default: 4)")
    parser.add_argument("--lr",         type=float, default=1e-5,
                        help="Learning rate (default: 1e-5)")
    parser.add_argument("--max_length", type=int,   default=256,
                        help="Max token length (default: 256)")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length
    )