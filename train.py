import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "indobenchmark/indobert-base-p1"
# Kalau laptop berat, ganti ke:
# MODEL_NAME = "indobenchmark/indobert-lite-base-p1"

DATA_PATH = "data/dataset_chatbot_it_support_client_netone.csv"
OUTPUT_DIR = "model-indobert-ticketing"


def split_dataset(df):
    label_counts = df["label_id"].value_counts()
    test_size = 0.2
    stratify_column = df["label_id"]

    if label_counts.min() < 2:
        print(
            "Peringatan: ada label dengan kurang dari 2 data. "
            "Split dilakukan tanpa stratify."
        )
        stratify_column = None
    elif int(np.ceil(len(df) * test_size)) < len(label_counts):
        test_size = len(label_counts) / len(df)

    return train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=stratify_column,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"])

    labels = sorted(df["label"].unique())
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}

    df["label_id"] = df["label"].map(label2id)

    print("Daftar label:")
    for label, index in label2id.items():
        print(f"{index}: {label}")

    train_df, test_df = split_dataset(df)

    train_dataset = Dataset.from_pandas(
        train_df[["text", "label_id"]].rename(columns={"label_id": "labels"})
    )

    test_dataset = Dataset.from_pandas(
        test_df[["text", "label_id"]].rename(columns={"label_id": "labels"})
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_steps=10,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    eval_result = trainer.evaluate()
    print("\nHasil evaluasi:")
    print(eval_result)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, "label_mapping.json"), "w") as file:
        json.dump(
            {
                "label2id": label2id,
                "id2label": id2label,
            },
            file,
            indent=4,
        )

    print(f"\nModel berhasil disimpan ke folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
