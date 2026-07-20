"""
train.py

Consolidated training script for the AI Learning Assistant emotion detection models.

This script documents and reproduces the pipeline originally run on Kaggle
(GPU-accelerated environment) to train:
  1. A BiLSTM baseline model + domain-adaptive fine-tuned version
  2. A fine-tuned DistilBERT transformer

NOTE: Training BERT and BiLSTM from scratch is compute-intensive. Running this
script locally without a GPU will be very slow. It is provided for
reproducibility and documentation purposes; the actual trained model files
used by the app were produced on Kaggle and are loaded directly by
src/model.py and src/bert_model.py.

Usage:
    python src/train.py --stage bilstm
    python src/train.py --stage bert
    python src/train.py --stage all
"""

import os
import re
import argparse
import random
import pickle
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MAX_VOCAB_SIZE = 30000
MAX_SEQ_LEN = 80
EMOTION_CLASSES = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]


# ----------------------------------------------------------------------
# Shared preprocessing
# ----------------------------------------------------------------------

def setup_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)


def clean_text(text, stopword_set):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stopword_set and len(t) > 1]
    return " ".join(tokens)


def generate_template_examples(templates_dict, subjects, n_per_class=440):
    """Generates synthetic examples for underrepresented classes (e.g. 'Bored')."""
    rows = []
    for emotion, templates in templates_dict.items():
        combos = [(s, t) for s in subjects for t in templates]
        random.shuffle(combos)
        for i in range(n_per_class):
            s, t = combos[i % len(combos)]
            sentence = t.format(s=s, S=s.capitalize())
            rows.append({"text": sentence, "emotion": emotion})
    return pd.DataFrame(rows)


def build_combined_dataset(data_dir="data"):
    """
    Combines and remaps source datasets (GoEmotions, ISEAR, EmoContext,
    Empathetic Dialogues) plus synthetic examples into the 5 target classes.

    NOTE: This assumes source datasets are downloaded under `data_dir` with
    the same structure used during the original Kaggle run. See
    PROJECT_ANALYSIS_REPORT.md Section 2 for the exact source-to-target
    label mapping used.
    """
    raise NotImplementedError(
        "Dataset combination logic is dataset-path specific and was run "
        "interactively on Kaggle. See PROJECT_ANALYSIS_REPORT.md Section 2 "
        "for the full mapping table, or load a previously saved "
        "'combined_preprocessed.csv' directly instead."
    )


# ----------------------------------------------------------------------
# BiLSTM training
# ----------------------------------------------------------------------

def train_bilstm(combined_csv="combined_preprocessed.csv", output_dir="models/bltsm"):
    import keras
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    import tensorflow as tf

    setup_nltk()
    stopword_set = set(stopwords.words("english"))

    print("Loading combined dataset...")
    combined_df = pd.read_csv(combined_csv)

    print("Cleaning text...")
    combined_df["clean_text"] = combined_df["text"].apply(lambda t: clean_text(t, stopword_set))

    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(combined_df["clean_text"])
    sequences = tokenizer.texts_to_sequences(combined_df["clean_text"])
    padded = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(combined_df["emotion"])

    X_train, X_temp, y_train, y_temp = train_test_split(
        padded, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    def focal_loss(gamma=2.0, alpha=0.25):
        def loss_fn(y_true, y_pred):
            y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), depth=len(label_encoder.classes_))
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
            cross_entropy = -y_true_oh * tf.math.log(y_pred)
            weight = alpha * tf.pow(1 - y_pred, gamma)
            return tf.reduce_sum(weight * cross_entropy, axis=-1)
        return loss_fn

    model = keras.Sequential([
        keras.layers.Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=128, mask_zero=True),
        keras.layers.Bidirectional(keras.layers.LSTM(128, dropout=0.2, use_cudnn=False)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(len(label_encoder.classes_), activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0),
        loss=focal_loss(gamma=2.0),
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
    ]

    print("Training BiLSTM...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=512,
        callbacks=callbacks,
        verbose=1,
    )

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, digits=4))

    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, "bilstm_student_adaptive.keras"))
    with open(os.path.join(output_dir, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
    with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"BiLSTM model saved to {output_dir}/")
    return model, tokenizer, label_encoder


# ----------------------------------------------------------------------
# BERT training
# ----------------------------------------------------------------------

def train_bert(combined_csv="combined_preprocessed.csv", output_dir="models/bert_emotion_model_final",
                max_per_class=3000, num_epochs=3):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    from tqdm.auto import tqdm
    from sklearn.metrics import accuracy_score, f1_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    combined_df = pd.read_csv(combined_csv)
    label_encoder = LabelEncoder()
    label_encoder.fit(combined_df["emotion"])

    # Balanced sampling for training efficiency
    sample_df = combined_df.groupby("emotion", group_keys=False).apply(
        lambda x: x.sample(min(len(x), max_per_class), random_state=42)
    ).reset_index(drop=True)

    texts = sample_df["text"].tolist()
    labels = label_encoder.transform(sample_df["emotion"])

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    class EmotionDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    def encode(txts):
        return tokenizer(txts, truncation=True, padding=True, max_length=MAX_SEQ_LEN)

    train_loader = DataLoader(EmotionDataset(encode(train_texts), train_labels), batch_size=32, shuffle=True)
    val_loader = DataLoader(EmotionDataset(encode(val_texts), val_labels), batch_size=64)
    test_loader = DataLoader(EmotionDataset(encode(test_texts), test_labels), batch_size=64)

    id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
    label2id = {label: i for i, label in enumerate(label_encoder.classes_)}

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label_encoder.classes_),
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    print("Training BERT...")
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} - train"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(batch["labels"].cpu().numpy())

        print(f"Epoch {epoch+1}: train_loss={np.mean(train_losses):.4f}, "
              f"val_acc={accuracy_score(val_true, val_preds):.4f}, "
              f"val_f1_macro={f1_score(val_true, val_preds, average='macro'):.4f}")

    # Final test evaluation
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(batch["labels"].cpu().numpy())

    print(classification_report(test_true, test_preds, target_names=label_encoder.classes_, digits=4))

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"BERT model saved to {output_dir}/")
    return model, tokenizer


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train emotion detection models.")
    parser.add_argument(
        "--stage", choices=["bilstm", "bert", "all"], default="all",
        help="Which model(s) to train."
    )
    parser.add_argument(
        "--data", default="combined_preprocessed.csv",
        help="Path to the combined, preprocessed dataset CSV."
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"'{args.data}' not found. Run the data preparation steps described in "
            "PROJECT_ANALYSIS_REPORT.md Section 2 first, or download the dataset "
            "from the original Kaggle notebook output."
        )

    if args.stage in ("bilstm", "all"):
        train_bilstm(combined_csv=args.data)

    if args.stage in ("bert", "all"):
        train_bert(combined_csv=args.data)