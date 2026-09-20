import math
from collections import defaultdict

class NaiveBayesClassifier:
    """Multinomial Naive Bayes classifier with Laplace smoothing."""
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_priors = {}
        self.word_counts = {}
        self.class_totals = {}
        self.vocab = set()
        self.classes = []

    def fit(self, tokenized_texts: list[list[str]], labels: list[str]):
        self.classes = list(set(labels))
        total_samples = len(labels)

        # Initialize tracking dictionaries
        for c in self.classes:
            self.class_priors[c] = 0.0
            self.word_counts[c] = defaultdict(int)
            self.class_totals[c] = 0

        # Count frequencies
        for tokens, label in zip(tokenized_texts, labels):
            self.class_priors[label] += 1
            for word in tokens:
                self.vocab.add(word)
                self.word_counts[label][word] += 1
                self.class_totals[label] += 1

        # Calculate prior probabilities P(c)
        for c in self.classes:
            self.class_priors[c] = self.class_priors[c] / total_samples

    def predict_single(self, tokens: list[str]) -> str:
        vocab_size = len(self.vocab)
        best_class = None
        best_log_prob = -float("inf")

        for c in self.classes:
            log_prob = math.log(self.class_priors[c])
            denominator = self.class_totals[c] + self.alpha * vocab_size

            for word in tokens:
                if word in self.vocab:
                    count = self.word_counts[c].get(word, 0)
                    prob = (count + self.alpha) / denominator
                    log_prob += math.log(prob)

            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_class = c

        return best_class

    def predict(self, tokenized_texts: list[list[str]]) -> list[str]:
        return [self.predict_single(tokens) for tokens in tokenized_texts]