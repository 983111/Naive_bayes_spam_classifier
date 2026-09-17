import pandas as pd

from sklearn.model_selection import train_test_split

from model.preprocess import clean_text
from model.naive_bayes import NaiveBayesSpam
from model.utils import create_wordcloud

df = pd.read_csv("dataset/spam.csv", encoding="latin1")

df = df[["label","text"]]

df["tokens"] = df["text"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["tokens"],
    df["label"],
    test_size=0.2,
    random_state=42
)

model = NaiveBayesSpam()

model.fit(X_train.tolist(), y_train.tolist())

print("Training complete.")

create_wordcloud(model.spam_counts, "Spam Vocabulary")
create_wordcloud(model.ham_counts, "Ham Vocabulary")