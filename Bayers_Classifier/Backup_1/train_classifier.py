"""
train_classifier.py

Trains a Multinomial Naive Bayes classifier on the developer-profile CSV
(produced by json_to_csv.py + generate_synthetic.py).

Feature engineering
───────────────────
Each row is treated as a bag-of-tokens: every non-empty language name and
library name is extracted and used as a binary feature token.
The vocabulary is built from the training data; unseen tokens at inference
time are simply ignored (zero count), which is the correct Bayesian behaviour.

Why Multinomial Naive Bayes?
────────────────────────────
  - Naturally handles discrete, sparse count features (token presence).
  - Works well with small labelled datasets augmented by pseudo-labels.
  - Produces calibrated class-posterior probabilities directly from Bayes' theorem.
  - Backed by Nigam et al. (2000) for exactly this kind of text/token classification.

Output
──────
  models/developer_classifier.joblib   ← trained pipeline (vectorizer + NB)
  models/classifier_report.txt         ← per-class metrics on held-out test set

Usage
──────
    python train_classifier.py                             # uses synthetic.csv
    python train_classifier.py --input synthetic.csv
    python train_classifier.py --input features_with_labels.csv --test-size 0.25
"""

import os
import sys
import csv
import json
import argparse
from typing import List, Dict, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

# ── Column names ──────────────────────────────────────────────────────────────

LANG_COLS = ['top1lang', 'top2lang', 'top3lang', 'top4lang', 'top5lang']
LIB_COLS  = ['top1lib',  'top2lib',  'top3lib',  'top4lib',  'top5lib',  'top6lib', 'top7lib']
LABEL_COL = 'category'

# ── Hyperparameters ───────────────────────────────────────────────────────────
# alpha: Laplace smoothing.  1.0 is the Bayesian standard prior.
# Lower values (0.1) make the model more aggressive with rare tokens.
ALPHA = 1.0


# ===========================================================================
# Feature engineering
# ===========================================================================

def row_to_token_string(row: Dict[str, str]) -> str:
    """
    Convert one CSV row into a single whitespace-separated token string
    that the CountVectorizer can process.

    Each language or library name is normalised to a single alphanumeric
    token so the vectorizer treats 'react-native' and 'reactnative' the same
    way and 'SwiftUI' and 'swiftui' are folded together.

    Languages get a 'LANG__' prefix and libraries get a 'LIB__' prefix so
    the same word appearing as both a language and a library doesn't collapse
    into one ambiguous token.
    """
    tokens = []
    for col in LANG_COLS:
        val = row.get(col, '').strip()
        if val:
            # normalise: lowercase, replace non-alphanum with underscore
            clean = ''.join(c if c.isalnum() else '_' for c in val.lower()).strip('_')
            tokens.append(f'LANG__{clean}')
    for col in LIB_COLS:
        val = row.get(col, '').strip()
        if val:
            clean = ''.join(c if c.isalnum() else '_' for c in val.lower()).strip('_')
            tokens.append(f'LIB__{clean}')
    return ' '.join(tokens)


def load_dataset(csv_path: str) -> Tuple[List[str], List[str]]:
    """
    Load CSV and return (token_strings, labels).
    Rows with empty label or no tokens are skipped with a warning.
    """
    X, y = [], []
    skipped = 0
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if LABEL_COL not in (reader.fieldnames or []):
            sys.exit(
                f"ERROR: Column '{LABEL_COL}' not found in {csv_path}.\n"
                f"  Available columns: {reader.fieldnames}\n"
                f"  The CSV must have a 'category' column as the last header."
            )
        for row in reader:
            label = row.get(LABEL_COL, '').strip()
            if not label:
                skipped += 1
                continue
            tokens = row_to_token_string(row)
            if not tokens:
                skipped += 1
                continue
            X.append(tokens)
            y.append(label)

    if skipped:
        print(f"  [INFO] Skipped {skipped} rows with empty label or no tokens.")
    return X, y


# ===========================================================================
# Training
# ===========================================================================

def build_pipeline() -> Pipeline:
    """
    CountVectorizer (binary mode) + MultinomialNB.

    binary=True: each token is 0/1 (present/absent).
    Using raw counts also works but binary is more appropriate here since
    the same library appearing twice in a row is an artifact of the schema,
    not meaningful frequency information.
    """
    return Pipeline([
        ('vectorizer', CountVectorizer(
            analyzer   = 'word',
            binary     = True,          # presence, not frequency
            token_pattern = r'\S+',     # whitespace-separated tokens (no char filtering)
            lowercase  = False,         # already normalised in row_to_token_string
        )),
        ('nb', MultinomialNB(alpha=ALPHA)),
    ])


def train(csv_path: str, test_size: float, model_dir: str, seed: int) -> None:
    print(f"\n{'='*60}")
    print(f"  Developer Profile Classifier — Training")
    print(f"{'='*60}\n")

    # ── Load ──────────────────────────────────────────────────────────────
    print(f"Loading data from: {csv_path}")
    X, y = load_dataset(csv_path)
    print(f"  Samples loaded  : {len(X)}")

    classes, counts = np.unique(y, return_counts=True)
    print(f"  Classes         : {len(classes)}")
    for cls, cnt in sorted(zip(classes, counts), key=lambda x: -x[1]):
        print(f"    {cls:35s}  {cnt:4d} samples")
    print()

    # ── Train / test split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    print(f"Train set: {len(X_train)}  |  Test set: {len(X_test)}")

    # ── Build and fit ─────────────────────────────────────────────────────
    print("\nFitting Multinomial Naive Bayes pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    vocab_size = len(pipeline.named_steps['vectorizer'].vocabulary_)
    print(f"  Vocabulary size : {vocab_size} unique tokens")

    # ── Cross-validation (5-fold, stratified) ────────────────────────────
    print("\nRunning 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    print(f"  CV accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Fold scores : {[f'{s:.4f}' for s in cv_scores]}")

    # ── Held-out test evaluation ──────────────────────────────────────────
    print("\nEvaluating on held-out test set...")
    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"  Test accuracy : {acc:.4f}")

    report = classification_report(y_test, y_pred, zero_division=0)
    print(f"\nClassification report:\n{report}")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(model_dir, exist_ok=True)
    model_path  = os.path.join(model_dir, 'developer_classifier.joblib')
    report_path = os.path.join(model_dir, 'classifier_report.txt')

    # Bundle everything the inference script needs
    bundle = {
        'pipeline':     pipeline,      # fitted Pipeline (vectorizer + NB)
        'classes':      list(pipeline.classes_),
        'vocab_size':   vocab_size,
        'cv_mean':      float(cv_scores.mean()),
        'cv_std':       float(cv_scores.std()),
        'test_accuracy': float(acc),
        'train_size':   len(X_train),
        'test_size_n':  len(X_test),
        'alpha':        ALPHA,
    }
    joblib.dump(bundle, model_path)
    print(f"\nModel saved → {os.path.abspath(model_path)}")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Developer Classifier — Training Report\n")
        f.write(f"{'='*50}\n")
        f.write(f"Input CSV      : {csv_path}\n")
        f.write(f"Train samples  : {len(X_train)}\n")
        f.write(f"Test samples   : {len(X_test)}\n")
        f.write(f"Vocabulary     : {vocab_size} tokens\n")
        f.write(f"Alpha (smooth) : {ALPHA}\n")
        f.write(f"CV accuracy    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
        f.write(f"Test accuracy  : {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred, labels=list(classes))))
    print(f"Report saved → {os.path.abspath(report_path)}")
    print(f"\n{'='*60}\n")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description='Train a Multinomial Naive Bayes developer profile classifier.'
    )
    p.add_argument('--input',     '-i', default='synthetic.csv',
                   help='CSV with category column. Default: synthetic.csv')
    p.add_argument('--test-size', type=float, default=0.20,
                   help='Fraction held out for evaluation. Default: 0.20')
    p.add_argument('--model-dir', default='models',
                   help='Directory where the model is saved. Default: models/')
    p.add_argument('--seed',      type=int, default=42)
    args = p.parse_args()
    train(args.input, args.test_size, args.model_dir, args.seed)

if __name__ == '__main__':
    main()
