from collections import Counter
import math

class NaiveBayesSpam:

    def __init__(self):
        self.spam_counts = Counter()
        self.ham_counts = Counter()

        self.spam_messages = 0
        self.ham_messages = 0

        self.spam_words = 0
        self.ham_words = 0

        self.vocabulary = set()

    # --------------------------
    # Training
    # --------------------------
    def fit(self, texts, labels):

        for words, label in zip(texts, labels):

            if label == "spam":
                self.spam_messages += 1

                for word in words:
                    self.spam_counts[word] += 1
                    self.spam_words += 1
                    self.vocabulary.add(word)

            else:
                self.ham_messages += 1

                for word in words:
                    self.ham_counts[word] += 1
                    self.ham_words += 1
                    self.vocabulary.add(word)

        self.total_messages = self.spam_messages + self.ham_messages
        self.vocab_size = len(self.vocabulary)

    # --------------------------
    # Priors
    # --------------------------
    def prior_spam(self):
        return self.spam_messages / self.total_messages

    def prior_ham(self):
        return self.ham_messages / self.total_messages

    # --------------------------
    # Likelihoods with Laplace
    # --------------------------
    def likelihood_spam(self, word):
        return (
            self.spam_counts[word] + 1
        ) / (
            self.spam_words + self.vocab_size
        )

    def likelihood_ham(self, word):
        return (
            self.ham_counts[word] + 1
        ) / (
            self.ham_words + self.vocab_size
        )

    # --------------------------
    # Predict single message
    # --------------------------
    def predict(self, words):

        spam_score = math.log(self.prior_spam())
        ham_score = math.log(self.prior_ham())

        for word in words:
            spam_score += math.log(self.likelihood_spam(word))
            ham_score += math.log(self.likelihood_ham(word))

        if spam_score > ham_score:
            return "spam"

        return "ham"

    # --------------------------
    # Batch prediction
    # --------------------------
    def predict_batch(self, texts):
        predictions = []

        for words in texts:
            predictions.append(self.predict(words))

        return predictions

    # --------------------------
    # Explain prediction
    # --------------------------
    def explain(self, words):

        explanation = []

        for word in words:
            explanation.append({
                "word": word,
                "spam_prob": self.likelihood_spam(word),
                "ham_prob": self.likelihood_ham(word)
            })

        return sorted(
            explanation,
            key=lambda x: x["spam_prob"] / x["ham_prob"],
            reverse=True
        )