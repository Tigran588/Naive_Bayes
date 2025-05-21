import pandas as pd
import re
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.split()

df = pd.read_csv("./spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)


def train_naive_bayes(texts, labels):
    spam_words = Counter()
    ham_words = Counter()
    spam_total = 0
    ham_total = 0

    for text, label in zip(texts, labels):
        words = preprocess(text)
        if label == 1:
            spam_words.update(words)
            spam_total += len(words)
        else:
            ham_words.update(words)
            ham_total += len(words)

    vocabulary = set(spam_words.keys()) | set(ham_words.keys())
    return spam_words, ham_words, spam_total, ham_total, vocabulary

# Calculate priors and likelihoods
def calculate_probs(spam_words, ham_words, spam_total, ham_total, vocabulary, alpha=1):
    vocab_size = len(vocabulary)
    p_word_spam = {}
    p_word_ham = {}
    for word in vocabulary:
        p_word_spam[word] = (spam_words.get(word, 0) + alpha) / (spam_total + alpha * vocab_size)
        p_word_ham[word] = (ham_words.get(word, 0) + alpha) / (ham_total + alpha * vocab_size)
    return p_word_spam, p_word_ham


def predict(text, p_word_spam, p_word_ham, p_spam, p_ham, vocabulary):
    words = preprocess(text)
    log_prob_spam = np.log(p_spam)
    log_prob_ham = np.log(p_ham)

    for word in words:
        if word in vocabulary:
            log_prob_spam += np.log(p_word_spam.get(word, 1e-6))
            log_prob_ham += np.log(p_word_ham.get(word, 1e-6))

    return 1 if log_prob_spam > log_prob_ham else 0

# TRAINING
spam_words, ham_words, spam_total, ham_total, vocabulary = train_naive_bayes(X_train, y_train)
p_spam = sum(y_train) / len(y_train)
p_ham = 1 - p_spam
p_word_spam, p_word_ham = calculate_probs(spam_words, ham_words, spam_total, ham_total, vocabulary)

#TEST
predictions = [predict(text, p_word_spam, p_word_ham, p_spam, p_ham, vocabulary) for text in X_test]
accuracy = np.mean(np.array(predictions) == y_test.values)

print(f"Accuracy on spam classification: {accuracy:.4f}")
