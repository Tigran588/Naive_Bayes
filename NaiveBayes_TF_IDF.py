import os
import re
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split


def load_imdb_data(data_dir, subset='train', max_docs=1000):
    texts, labels = [], []
    for label in ['pos', 'neg']:
        folder = os.path.join(data_dir, subset, label)
        for i, filename in enumerate(os.listdir(folder)):
            if i >= max_docs:
                break
            with open(os.path.join(folder, filename), encoding='utf-8') as f:
                texts.append(f.read())
                labels.append('pos' if label == 'pos' else 'neg')
    return texts, labels


def preprocess_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', '', text.lower()) 
    return text.split()


def compute_tf(documents):
    tf_scores = []
    for doc in documents:
        word_counts = Counter(doc)
        total_words = max(len(doc), 1)
        tf_doc = {word: count / total_words for word, count in word_counts.items()}
        tf_scores.append(tf_doc)
    return tf_scores

def compute_idf(documents, vocabulary):
    N = len(documents)
    idf_scores = {}
    for word in vocabulary:
        doc_count = sum(1 for doc in documents if word in doc)
        idf_scores[word] = np.log((N + 1) / (doc_count + 1)) + 1
    return idf_scores

def compute_tfidf(documents, vocabulary, tf_scores, idf_scores):
    tfidf_matrix = []
    for tf_doc in tf_scores:
        tfidf_doc = {word: tf_doc.get(word, 0) * idf_scores.get(word, 0) for word in vocabulary}
        tfidf_matrix.append(tfidf_doc)
    return tfidf_matrix

# N_BAYES TRAINING
def calculate_priors(labels):
    total_docs = len(labels)
    pos_count = sum(1 for label in labels if label == "pos")
    p_pos = pos_count / total_docs
    p_neg = (total_docs - pos_count) / total_docs
    return p_pos, p_neg

def calculate_conditional_probs(tfidf_matrix, labels, vocabulary, alpha=1):
    vocab_size = len(vocabulary)
    pos_word_counts = Counter()
    neg_word_counts = Counter()
    pos_total, neg_total = 0, 0

    for tfidf_doc, label in zip(tfidf_matrix, labels):
        for word, score in tfidf_doc.items():
            if label == "pos":
                pos_word_counts[word] += score
                pos_total += score
            else:
                neg_word_counts[word] += score
                neg_total += score

    pos_total = max(pos_total, 1e-10)
    neg_total = max(neg_total, 1e-10)

    p_word_given_pos = {word: (pos_word_counts.get(word, 0) + alpha) / (pos_total + alpha * vocab_size) for word in vocabulary}
    p_word_given_neg = {word: (neg_word_counts.get(word, 0) + alpha) / (neg_total + alpha * vocab_size) for word in vocabulary}
    return p_word_given_pos, p_word_given_neg, pos_total, neg_total

def predict(text, vocabulary, p_pos, p_neg, p_word_given_pos, p_word_given_neg, idf_scores, pos_total, neg_total):
    words = preprocess_text(text)
    tf_scores = Counter(words)
    total_words = max(len(words), 1)
    tfidf_doc = {word: (tf_scores.get(word, 0) / total_words) * idf_scores.get(word, 0) for word in vocabulary}

    log_p_pos = np.log(p_pos)
    log_p_neg = np.log(p_neg)

    for word in vocabulary:
        if tfidf_doc[word] > 0:
            log_p_pos += tfidf_doc[word] * np.log(p_word_given_pos.get(word, 1 / (pos_total + len(vocabulary))))
            log_p_neg += tfidf_doc[word] * np.log(p_word_given_neg.get(word, 1 / (neg_total + len(vocabulary))))

    return "pos" if log_p_pos > log_p_neg else "neg"


if __name__ == "__main__":
 
    train_texts, train_labels = load_imdb_data("aclImdb", "train", max_docs=1000)
    test_texts, test_labels = load_imdb_data("aclImdb", "test", max_docs=200)

    # Preprocess
    train_tokens = [preprocess_text(text) for text in train_texts]
    test_tokens = [preprocess_text(text) for text in test_texts]


    vocabulary = list(set(word for doc in train_tokens for word in doc))

    # TF-IDF
    tf_scores_train = compute_tf(train_tokens)
    idf_scores = compute_idf(train_tokens, vocabulary)
    tfidf_matrix_train = compute_tfidf(train_tokens, vocabulary, tf_scores_train, idf_scores)

    # N_BATES TRAIN
    p_pos, p_neg = calculate_priors(train_labels)
    p_word_given_pos, p_word_given_neg, pos_total, neg_total = calculate_conditional_probs(tfidf_matrix_train, train_labels, vocabulary)


    correct = 0
    for text, true_label in zip(test_texts, test_labels):
        pred = predict(text, vocabulary, p_pos, p_neg, p_word_given_pos, p_word_given_neg, idf_scores, pos_total, neg_total)
        correct += (pred == true_label)

    accuracy = correct / len(test_labels)
    print(f"Test Accuracy on {len(test_labels)} reviews: {accuracy:.3f}")
