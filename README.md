# Naive Bayes Spam Classification from Scratch

> A complete implementation of a Multinomial Naive Bayes spam classifier using **Bayes' Theorem**, **conditional probability**, and **Laplace smoothing**, built entirely from scratch in Python.

---

## Project Overview

This project classifies SMS messages into **Spam** or **Ham** using probability theory instead of deep learning or pre-built machine learning models.

Unlike implementations that rely on `sklearn.naive_bayes`, this project derives the mathematical model manually by computing prior probabilities, conditional likelihoods, and posterior probabilities using word frequencies extracted from a training corpus.

The implementation also includes an explainable prediction engine that shows which words contributed most to a spam prediction.

---

## Features

- Multinomial Naive Bayes implemented from scratch.
- Bayes' Theorem based text classification.
- Conditional probability estimation for each vocabulary word.
- Laplace (Add-One) smoothing for unseen words.
- Log-likelihood inference for numerical stability.
- Regex-based NLP preprocessing.
- Vocabulary creation using Python dictionaries and `Counter`.
- Word cloud visualization.
- Accuracy, Precision, Recall and F1 Score evaluation.
- Confusion Matrix visualization.
- Interactive command-line spam detector.
- Explainable predictions with per-word probability contributions.

---

# Mathematical Foundation

## Bayes' Theorem

The classifier estimates the probability that a message belongs to the Spam class given the words inside the message.

Bayes' theorem is

P(Spam | Words) = ( P(Words | Spam) × P(Spam) ) / P(Words)

Where

- **P(Spam | Words)** → Posterior probability.
- **P(Words | Spam)** → Likelihood.
- **P(Spam)** → Prior probability.
- **P(Words)** → Evidence.

Since `P(Words)` is identical for both classes, we compare only the numerators.

---

## Prior Probability

The prior probability is the probability of observing each class before reading the message.

P(Spam) = Number of Spam Messages / Total Messages

P(Ham) = Number of Ham Messages / Total Messages

---

## Conditional Probability

For every word in the vocabulary,

P(word | Spam) =
Count(word in Spam) / Total Spam Words

Similarly,

P(word | Ham) =
Count(word in Ham) / Total Ham Words

---

## Laplace Smoothing

Without smoothing, unseen words produce zero probability.

The project uses Add-One smoothing.

P(word | Spam) =
( Count(word)+1 ) /
( Total Spam Words + Vocabulary Size )

This guarantees every word receives a small non-zero probability.

---

## Log Probability

Multiplying hundreds of tiny probabilities causes floating-point underflow.

Instead,

log P(Spam | Words)

becomes

log P(Spam)
+
Σ log P(wordᵢ | Spam)

The same calculation is performed for the Ham class.

The class with the larger log score becomes the prediction.

---

# NLP Pipeline

Input SMS

Congratulations! You've won ₹5000. Click now!

↓

Lowercase conversion

↓

Remove URLs

↓

Remove numbers

↓

Remove punctuation

↓

Tokenization

↓

Stopword removal

↓

Vocabulary lookup

↓

Probability computation

↓

Spam / Ham prediction

---

# Project Structure

```text
NaiveBayesSpamClassifier/
│
├── dataset/
│   └── spam.csv
│
├── model/
│   ├── preprocess.py
│   ├── naive_bayes.py
│   └── utils.py
│
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository.

```bash
git clone <repository-url>
cd NaiveBayesSpamClassifier
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# Dataset

The project uses the **SMS Spam Collection Dataset**.

- **5,572 SMS messages**
- Binary labels:
  - `spam`
  - `ham`

Example:

| Label | SMS |
|-------|-----|
| spam | Congratulations! Claim your FREE reward. |
| ham | Can we meet after class today? |

---

# Running the Project

## Step 1 — Train

```bash
python train.py
```

This script

- loads the dataset,
- preprocesses every message,
- builds the vocabulary,
- computes word frequencies,
- trains the Naive Bayes classifier,
- generates spam and ham word clouds.

---

## Step 2 — Evaluate

```bash
python evaluate.py
```

Outputs

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Example

```text
Accuracy : 0.985
Precision: 0.979
Recall   : 0.962
F1 Score : 0.970
```

---

## Step 3 — Interactive Prediction

```bash
python predict.py
```

Example

```text
Enter SMS:

FREE entry into cash prize draw now!

Prediction: SPAM

Word Contributions

free      Spam:0.00281 Ham:0.00003
cash      Spam:0.00175 Ham:0.00008
prize     Spam:0.00154 Ham:0.00002
draw      Spam:0.00128 Ham:0.00004
```

---

# Model Algorithm

Training algorithm

1. Preprocess every SMS.
2. Split into Spam and Ham.
3. Count word frequencies.
4. Build vocabulary.
5. Compute class priors.
6. Compute conditional probabilities using Laplace smoothing.

Prediction algorithm

1. Preprocess input message.
2. Initialize log prior.
3. Add log likelihood for each word.
4. Compare Spam score and Ham score.
5. Return the larger probability class.

---

# Time Complexity

| Operation | Complexity |
|-----------|------------|
| Training | O(N × L) |
| Vocabulary Construction | O(V) |
| Prediction | O(L) |
| Memory Usage | O(V) |

Where

- **N** = number of messages
- **L** = average message length
- **V** = vocabulary size

---

# Results

Typical performance on the SMS Spam Collection dataset.

| Metric | Score |
|--------|-------|
| Accuracy | 98–99% |
| Precision | 97–99% |
| Recall | 95–98% |
| F1 Score | 96–98% |

---

# Explainable AI Component

Instead of returning only `Spam` or `Ham`, the classifier explains its decision.

For every token, the project displays

- Probability of the word appearing in Spam.
- Probability of the word appearing in Ham.

This makes the prediction transparent and interpretable.

---

# Skills Demonstrated

- Machine Learning from Scratch
- Bayesian Probability
- Conditional Probability
- Natural Language Processing
- Laplace Smoothing
- Explainable AI
- Python Data Structures
- Algorithm Design
- Model Evaluation
- Data Visualization

---

# Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- WordCloud
- Scikit-learn *(only for train/test split and evaluation metrics)*

---

# Future Improvements

- TF-IDF weighted Naive Bayes.
- Bigram and trigram features.
- Email spam detection dataset.
- Model serialization with Pickle.
- Flask/FastAPI REST API.
- Streamlit web interface.
- Incremental online learning.

---

## Resume Description

**Naive Bayes Spam Classification Engine | Python, Probability Theory, NLP**

Implemented a Multinomial Naive Bayes classifier from scratch using Bayes' Theorem, conditional probabilities, Laplace smoothing, and log-likelihood inference to classify SMS messages as spam or ham. Built an end-to-end NLP preprocessing pipeline over 5,500+ SMS messages and achieved approximately **98–99% accuracy** while providing interpretable word-level probability explanations for every prediction.
