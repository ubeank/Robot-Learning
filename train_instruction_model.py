import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_CSV = SCRIPT_DIR / "text_train_36400_edited.csv"
DEFAULT_EVAL_CSV = SCRIPT_DIR / "text_eval_600_edited.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "models" / "instruction_bert"

VALID_PLACES = ["bathroom", "bedroom", "kitchen", "living_room"]
VALID_DIFFICULTIES = ["ambiguous", "direct", "indirect"]


def normalize_place(value):
    return str(value).strip().replace(" ", "_")


def read_instruction_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"sentence", "class", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df[["sentence", "class", "label"]].dropna().copy()
    df["sentence"] = df["sentence"].astype(str)
    df["class"] = df["class"].map(normalize_place)
    df["label"] = df["label"].astype(str).str.strip()

    bad_places = sorted(set(df["class"]) - set(VALID_PLACES))
    bad_difficulties = sorted(set(df["label"]) - set(VALID_DIFFICULTIES))
    if bad_places:
        raise ValueError(f"Unsupported place labels in {path}: {bad_places}")
    if bad_difficulties:
        raise ValueError(f"Unsupported difficulty labels in {path}: {bad_difficulties}")

    return df


class InstructionDataset(Dataset):
    def __init__(self, df, tokenizer, place_to_id, difficulty_to_id, max_length):
        self.sentences = df["sentence"].tolist()
        self.place_ids = [place_to_id[x] for x in df["class"].tolist()]
        self.difficulty_ids = [difficulty_to_id[x] for x in df["label"].tolist()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.sentences[index],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["place_id"] = torch.tensor(self.place_ids[index], dtype=torch.long)
        item["difficulty_id"] = torch.tensor(
            self.difficulty_ids[index],
            dtype=torch.long,
        )
        return item


class MultiTaskInstructionModel(nn.Module):
    def __init__(self, encoder, hidden_size, num_places, num_difficulties):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.1)
        self.place_head = nn.Linear(hidden_size, num_places)
        self.difficulty_head = nn.Linear(hidden_size, num_difficulties)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return self.place_head(pooled), self.difficulty_head(pooled)


def move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def run_eval(model, loader, device):
    model.eval()
    place_true = []
    place_pred = []
    difficulty_true = []
    difficulty_pred = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            place_logits, difficulty_logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
            )
            place_true.extend(batch["place_id"].cpu().tolist())
            difficulty_true.extend(batch["difficulty_id"].cpu().tolist())
            place_pred.extend(place_logits.argmax(dim=-1).cpu().tolist())
            difficulty_pred.extend(difficulty_logits.argmax(dim=-1).cpu().tolist())

    return {
        "place_acc": accuracy_score(place_true, place_pred),
        "difficulty_acc": accuracy_score(difficulty_true, difficulty_pred),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--eval_csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    train_df = read_instruction_csv(args.train_csv)
    eval_df = read_instruction_csv(args.eval_csv)

    place_labels = sorted(train_df["class"].unique().tolist())
    difficulty_labels = sorted(train_df["label"].unique().tolist())
    place_to_id = {label: idx for idx, label in enumerate(place_labels)}
    difficulty_to_id = {label: idx for idx, label in enumerate(difficulty_labels)}

    print(f"Train rows: {len(train_df)}")
    print(f"Eval rows : {len(eval_df)}")
    print(f"Places    : {place_labels}")
    print(f"Difficulty: {difficulty_labels}")
    print(f"Model     : {args.model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            local_files_only=args.local_files_only,
        )
        encoder = AutoModel.from_pretrained(
            args.model_name,
            local_files_only=args.local_files_only,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load the BERT/DistilBERT base model. "
            "Do not install torch into ithor_env. Use system python3, and make "
            "sure the model is downloaded or allow this script to download it. "
            f"Original error: {exc}"
        ) from exc

    train_ds = InstructionDataset(
        train_df,
        tokenizer,
        place_to_id,
        difficulty_to_id,
        args.max_length,
    )
    eval_ds = InstructionDataset(
        eval_df,
        tokenizer,
        place_to_id,
        difficulty_to_id,
        args.max_length,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskInstructionModel(
        encoder=encoder,
        hidden_size=encoder.config.hidden_size,
        num_places=len(place_labels),
        num_difficulties=len(difficulty_labels),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")

        for batch in progress:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()

            place_logits, difficulty_logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
            )
            place_loss = loss_fn(place_logits, batch["place_id"])
            difficulty_loss = loss_fn(difficulty_logits, batch["difficulty_id"])
            loss = place_loss + difficulty_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        metrics = run_eval(model, eval_loader, device)
        avg_loss = total_loss / max(len(train_loader), 1)
        print(
            f"epoch={epoch} loss={avg_loss:.4f} "
            f"place_acc={metrics['place_acc']:.4f} "
            f"difficulty_acc={metrics['difficulty_acc']:.4f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder_dir = args.output_dir / "encoder"
    model.encoder.save_pretrained(encoder_dir)
    tokenizer.save_pretrained(args.output_dir)

    torch.save(
        {
            "place_head": model.place_head.state_dict(),
            "difficulty_head": model.difficulty_head.state_dict(),
        },
        args.output_dir / "heads.pt",
    )

    label_maps = {
        "model_name": args.model_name,
        "encoder_dir": "encoder",
        "place_labels": place_labels,
        "difficulty_labels": difficulty_labels,
        "max_length": args.max_length,
    }
    with open(args.output_dir / "label_maps.json", "w", encoding="utf-8") as f:
        json.dump(label_maps, f, indent=2)

    print(f"Saved model to: {args.output_dir}")


if __name__ == "__main__":
    main()
