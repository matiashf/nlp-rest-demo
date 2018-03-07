import numpy as np
import pytest

import data
from model import make_model

@pytest.fixture(scope='session')
def word_vectors():
    return data.get_word_vectors(max_vocab_size=10, vector_size=50)

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

def test_train_test_generators(word_vectors):
    train, test = data.generators(train_ratio=3, test_ratio=1, batch_size=8,
                                  num_words=15, word_vectors=word_vectors)
    train_batch = next(iter(train))
    test_batch = next(iter(test))

def test_make_model(word_vectors):
    make_model(num_words=15, word_vectors=word_vectors)

def test_prepare(word_vectors):
    with pytest.raises(data.InvalidTextError):
        data.prepare('', word_vectors, num_words=3)

    # Get the indices of some common words
    the = word_vectors.indices['the']
    of = word_vectors.indices['of']
    pad = word_vectors.indices[None] # Zeroes are used for padding
    assert pad == 0

    # Are we pre-padding correctly?
    indices = data.prepare('the of', word_vectors, num_words=3)
    assert all(indices == (pad, the, of))

    # pre-truncating?
    indices = data.prepare('the the of of', word_vectors, num_words=3)
    assert all(indices == (the, of, of))
