import numpy as np

import data
from model import make_model

def test_samples():
    for i, (features, label) in enumerate(data.samples()):
        if i == 10:
            break

def test_split():
    indices = np.array([2, 4, 8, 9])
    expected = np.concatenate((indices, indices + 10))
    actual = np.array(list(data.split(indices, 10, range(20))))
    assert len(actual) == len(expected)
    assert actual.dtype == expected.dtype
    assert all(actual == expected)

def test_train_test_generators():
    word_vectors = data.get_word_vectors(max_vocab_size=10, vector_size=50)
    train, test = data.generators(train_ratio=3, test_ratio=1, batch_size=8,
                                  num_words=15, word_vectors=word_vectors)
    train_batch = next(iter(train))
    test_batch = next(iter(test))

def test_make_model():
    make_model(num_words=15, word_vector_size=50)
