from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from Backend.Pipeline.preprocessing import load_all_datasets, clean_combined, tokenize_and_save
from Backend.config import ARTIFACTS_DIR

EVAL_DIR = Path("Backend/Evaluation")
PLOTS_DIR = EVAL_DIR / "plots"
STATUS_PATH = EVAL_DIR / "status.json"
METRICS_PATH = EVAL_DIR / "metrics.json"

MODEL_NAME = "distilbert-base-uncased"


def write_status(stage: str, message: str, done: bool = False, error: str | None = None):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "message": message,
        "done": done,
        "error": error,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_confusion_matrix(y_true, y_pred, output_path: Path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["FAKE", "TRUE"])
    plt.yticks([0, 1], ["FAKE", "TRUE"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_roc_curve(y_true, y_score, output_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return roc_auc


def main():
    try:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        write_status("preprocessing", "Loading and preprocessing datasets...")

        df_all = load_all_datasets(
            include_kaggle=True,
            include_fever=True,
            include_pubhealth=True,
            include_social=True,
        )
        df_all = clean_combined(df_all)
        tokenize_and_save(df_all)

        write_status("training", "Loading tokenized datasets...")

        train_ds = load_from_disk(str(ARTIFACTS_DIR / "tokenized" / "train"))
        eval_ds = load_from_disk(str(ARTIFACTS_DIR / "tokenized" / "eval"))

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

        write_status("training", "Training DistilBERT model...")

        training_args = TrainingArguments(
            output_dir="artifacts/model",
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            num_train_epochs=3,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=32,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to="none",
        )


        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.save_model(str(ARTIFACTS_DIR / "model"))

        write_status("evaluation", "Evaluating model and generating plots...")

        predictions = trainer.predict(eval_ds)
        logits = predictions.predictions
        y_true = predictions.label_ids
        y_pred = np.argmax(logits, axis=1)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        y_score_true = probs[:, 1]

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label=1, zero_division=0
        )

        roc_auc = save_roc_curve(y_true, y_score_true, PLOTS_DIR / "roc_curve.png")
        save_confusion_matrix(y_true, y_pred, PLOTS_DIR / "confusion_matrix.png")

        metrics = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "train_size": len(train_ds),
            "eval_size": len(eval_ds),
            "last_updated": datetime.now().isoformat(timespec="seconds"),
            "plots": {
                "confusion_matrix": "/evaluation/plots/confusion_matrix.png",
                "roc_curve": "/evaluation/plots/roc_curve.png",
            },
        }

        METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        write_status("complete", "Training and evaluation complete.", done=True)

    except Exception as e:
        write_status("error", "Training/evaluation failed.", done=True, error=f"{type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()