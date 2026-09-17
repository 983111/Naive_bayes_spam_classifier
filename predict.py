import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.model_selection import train_test_split

from model.preprocess import clean_text
from model.naive_bayes import NaiveBayesSpam

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

predictions = model.predict_batch(X_test.tolist())

print("Accuracy :", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test,predictions,pos_label="spam"))
print("Recall   :", recall_score(y_test,predictions,pos_label="spam"))
print("F1 Score :", f1_score(y_test,predictions,pos_label="spam"))

cm = confusion_matrix(y_test,predictions,labels=["ham","spam"])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["ham","spam"]
)

disp.plot(cmap="Blues")
plt.show()