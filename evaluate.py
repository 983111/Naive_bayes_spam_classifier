import os
import random
import pandas as pd
from model.preprocess import clean_text
from model.naive_bayes import NaiveBayesClassifier

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, encoding="latin-1")
    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "text"})
    df = df[["label", "text"]].dropna()
    return df

def train_test_split_custom(records, test_ratio=0.2, seed=42):
    random.seed(seed)
    shuffled = records.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]

def main():
    dataset_path = os.path.join("dataset", "spam.csv")
    df = load_data(dataset_path)

    records = list(zip(df["text"].tolist(), df["label"].tolist()))
    train_data, test_data = train_test_split_custom(records, test_ratio=0.2)

    # Train on 80% split
    x_train_tokens = [clean_text(text) for text, _ in train_data]
    y_train = [label for _, label in train_data]

    model = NaiveBayesClassifier(alpha=1.0)
    model.fit(x_train_tokens, y_train)

    # Evaluate on remaining 20%
    x_test_tokens = [clean_text(text) for text, _ in test_data]
    y_test = [label for _, label in test_data]
    y_pred = model.predict(x_test_tokens)

    # Calculate metrics treating 'spam' as the positive class
    tp = sum(1 for yt, yp in zip(y_test, y_pred) if yt == "spam" and yp == "spam")
    fp = sum(1 for yt, yp in zip(y_test, y_pred) if yt != "spam" and yp == "spam")
    fn = sum(1 for yt, yp in zip(y_test, y_pred) if yt == "spam" and yp != "spam")
    tn = sum(1 for yt, yp in zip(y_test, y_pred) if yt != "spam" and yp != "spam")

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    print(f"Accuracy : {accuracy:.16f}")
    print(f"Precision: {precision:.16f}")
    print(f"Recall   : {recall:.16f}")
    print(f"F1 Score : {f1:.16f}")

if __name__ == "__main__":
    main()