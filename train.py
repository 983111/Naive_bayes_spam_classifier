import os
import pickle
import pandas as pd
from model.preprocess import clean_text
from model.naive_bayes import NaiveBayesClassifier

def load_data(filepath: str) -> pd.DataFrame:
    # Read CSV supporting latin-1 encoding common with SMS spam datasets
    df = pd.read_csv(filepath, encoding="latin-1")
    
    # Handle standard SMS collection format (v1=label, v2=text) or pre-labeled formats
    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "text"})
    
    df = df[["label", "text"]].dropna()
    return df

def main():
    dataset_path = os.path.join("dataset", "spam.csv")
    print(f"Loading data from {dataset_path}...")
    df = load_data(dataset_path)

    print("Cleaning text...")
    tokenized_texts = [clean_text(t) for t in df["text"]]
    labels = df["label"].tolist()

    print("Training Naive Bayes model...")
    model = NaiveBayesClassifier(alpha=1.0)
    model.fit(tokenized_texts, labels)

    output_model_path = "spam_classifier.pkl"
    with open(output_model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {output_model_path}")
    print("Training complete.")

if __name__ == "__main__":
    main()