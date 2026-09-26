"""Fine-tune DistilBERT with LoRA using LIAR train and validation splits."""

import argparse
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from training.prepare_lora_data import (
    DATA_DIR,
    DEFAULT_BATCH_SIZE,
    MAX_LENGTH,
    TOKENIZER_DIR,
    LiarTokenizedDataset,
)

SEED = 42
EPOCHS = 3
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "models" / "distilbert-liar-lora"


def evaluate(model, data_loader, device):
    """Measure validation loss and macro-F1 without updating model weights."""
    model.eval()
    total_loss = 0.0
    example_count = 0
    predictions = []
    labels = []

    with torch.no_grad():
        for batch in data_loader:
            batch = {name: values.to(device) for name, values in batch.items()}
            result = model(**batch)
            count = batch["labels"].size(0)
            total_loss += result.loss.item() * count
            example_count += count
            predictions.extend(result.logits.argmax(dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())

    return {
        "validation_loss": total_loss / example_count,
        "validation_macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    if not TOKENIZER_DIR.is_dir():
        raise FileNotFoundError(f"Local DistilBERT checkpoint not found: {TOKENIZER_DIR}")
    for split in ("train", "validation"):
        path = DATA_DIR / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Dataset split not found: {path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output folder: {output_dir}")

    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_data = LiarTokenizedDataset(DATA_DIR / "train.jsonl", tokenizer)
    validation_data = LiarTokenizedDataset(DATA_DIR / "validation.jsonl", tokenizer)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_data,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        TOKENIZER_DIR,
        local_files_only=True,
    )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)
    model.to(device)

    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(f"Device: {device}")
    print(f"Training claims: {len(train_data)}; validation claims: {len(validation_data)}")
    print(f"Batch size: {DEFAULT_BATCH_SIZE}; max length: {MAX_LENGTH}; epochs: {EPOCHS}")
    print(f"Trainable parameters: {trainable_count:,}")
    print("The test split will not be loaded or evaluated.")

    best_macro_f1 = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        train_example_count = 0
        for batch in train_loader:
            batch = {name: values.to(device) for name, values in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            result = model(**batch)
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                max_norm=1.0,
            )
            optimizer.step()
            count = batch["labels"].size(0)
            total_train_loss += result.loss.item() * count
            train_example_count += count

        metrics = evaluate(model, validation_loader, device)
        train_loss = total_train_loss / train_example_count
        print(json.dumps({
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
        }))

        if metrics["validation_macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["validation_macro_f1"]
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            (output_dir / "training_metadata.json").write_text(
                json.dumps({
                    "base_checkpoint": str(TOKENIZER_DIR),
                    "epochs_requested": EPOCHS,
                    "best_epoch": epoch,
                    "best_validation_macro_f1": best_macro_f1,
                    "batch_size": DEFAULT_BATCH_SIZE,
                    "max_length": MAX_LENGTH,
                    "truncation": True,
                    "padding": "dynamic",
                    "lora": {
                        "rank": lora_config.r,
                        "alpha": lora_config.lora_alpha,
                        "dropout": lora_config.lora_dropout,
                        "target_modules": sorted(lora_config.target_modules),
                    },
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "seed": SEED,
                    "test_split_used": False,
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Saved best adapter and tokenizer to {output_dir}")

    print(f"Training complete. Best validation macro-F1: {best_macro_f1:.4f}")
    print("The test split was not used; evaluate it separately later.")


if __name__ == "__main__":
    main()
