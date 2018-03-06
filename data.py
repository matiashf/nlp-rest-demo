import os
import zipfile
from copy import copy
from collections import namedtuple
from functools import partial

import pandas as pd
import numpy as np
import keras
from keras.preprocessing.text import text_to_word_sequence

FILENAME = os.path.join(os.path.dirname(__file__), 'yelp_review.csv.zip')
assert os.path.exists(FILENAME)
assert zipfile.is_zipfile(FILENAME)

def samples(chunksize=1024):
    """Yield tuples of (features, label) one sample at a time"""

    # Use pandas to read CSV in chunks; It is the fastest by far
    reader = pd.read_csv(FILENAME, usecols=('text', 'stars'),
                         dtype=dict(text=str, stars=np.uint8),
                         iterator=True, chunksize=chunksize,
                         compression='zip',
                         memory_map=True) # Blazingly fast!
    for chunk in reader:
        chunk = chunk[chunk.stars != 3] # Discard neutral reviews
        chunk['label'] = chunk.stars > 0 # Label: Positive sentiment (bool)
        yield from chunk[['text', 'label']].itertuples(index=False)

def num_samples(word_vectors):
    count = 0
    for text, _ in samples():
        count += any(w in word_vectors for w in text_to_word_sequence(text))
    return count

def split(indices, period, iterable):
    """Helper to train/test split on an iterator

    This function doesn't actually split into separate streams, but
    chooses elements matching the given indices for each period within
    a stream. To obtain separate streams for testing and training, two
    separate calls to split are made on two separate sample
    streams. This favors (significantly) less memory use at the cost
    of (slightly) more cpu time.
    """

    indices = sorted(indices)
    assert period >= indices[-1]

    i = 0 # Index into iterator % period
    j = 0 # Index into indices
    for element in iterable:
        if i == indices[j]:
            yield element
            j = (j + 1) % len(indices)
        i = (i + 1) % period

def prepare(text, word_vectors, num_words):
    """Turn a single text into a 2D word embedding matrix (a vector for each word)

    Raises ValueError if no word vectors could be found."""

    indices = np.zeros(num_words, dtype=np.uint32)
    # Extract word indices from text, which later can be used to look
    # up word vectors. We start from the back to get pre-padding and
    # pre-truncation. This plays (slightly) nicer with recurrent
    # models.
    i = 1 # Negative index into indices, in range [1, num_words]
    for word in reversed(text_to_word_sequence(text)):
        if i == num_words + 1:
            break # Pre-truncate the rest of words (there were too many)
        index = word_vectors.indices.get(word, 0)
        indices[-i] = index
        i += (index != 0) # Increment index if word was found
    if i == 1: # No word indices found?
        raise KeyError(f"No vocabulary words in {text!r}")
    else:
        return indices

def batch(iterator, batch_size, num_words, word_vectors):
    """Group together individual samples into batches of (features, labels)"""

    features = np.empty((batch_size, num_words), dtype=np.uint32)
    labels = np.empty(batch_size, dtype=np.uint8)

    i = 0
    for text, label in iterator:
        # Prepare word embeddings
        try:
            word_indices = prepare(text, word_vectors, num_words)
        except KeyError:
            continue # Sample did not contain any words in our vocabulary
        else:
            features[i] = word_indices

        labels[i] = label

        i += 1
        if i == batch_size:
            yield features.copy(), labels.copy() # Copy to avoid side-effects
            i = 0

    return features

def endless_batches(indices, period, *args, **kwargs):
    """Combine the samples, split and batch functions into one"""

    while True:
        yield from batch(split(indices, period, samples()), *args, **kwargs)

def generators(train_ratio, test_ratio, batch_size, num_words, word_vectors,
               seed=None):
    """Return a tuple of train and test generators suitable for passing to keras"""

    assert train_ratio >= 1
    assert test_ratio >= 1

    period = train_ratio + test_ratio
    random = np.random.RandomState(seed)
    train_indices = random.choice(period, size=train_ratio, replace=False)
    test_indices = set(range(period)) - set(train_indices)
    assert test_indices.union(train_indices) == set(range(period))

    generate = partial(endless_batches, period=period, batch_size=batch_size,
                       num_words=num_words, word_vectors=word_vectors)
    return generate(train_indices), generate(test_indices)

WordVectors = namedtuple('WordVectors', ('words', 'indices', 'vectors'))

word_vector_sizes = (50, 100, 200, 300)
def get_word_vectors(max_vocab_size, vector_size):
    assert max_vocab_size > 0
    assert vector_size in word_vector_sizes

    # get_file downloads and caches files in ~/.keras/datasets/
    path = keras.utils.get_file(
        'glove.6B.zip',
        origin='http://nlp.stanford.edu/data/glove.6B.zip',
        file_hash='617afb2fe6cbd085c235baf7a465b96f4112bd7f7ccb2b2cbd649fed9cbcf2fb',
        extract=False,
    )

    word_vectors = WordVectors(
        # For mapping index -> word (for debugging purposes)
         # dtype object because we don't know the string length beforehand
        words=np.empty((max_vocab_size + 1), dtype=object),
        # For mapping word -> index
        indices=dict(),
        # For mapping index -> vector
        vectors=np.empty((max_vocab_size + 1, vector_size),
                         dtype=np.float32)
    )

    # Index zero is used for words not in the vocabulary
    word_vectors.words[0] = None
    word_vectors.vectors[0] = 0
    word_vectors.indices[None] = 0

    with zipfile.ZipFile(path) as zf:
        i = 1
        for line in zf.open(f"glove.6B.{vector_size}d.txt"):
            if i == max_vocab_size + 1:
                break
            values = line.decode().split()
            word = values[0]

            if text_to_word_sequence(word) != [word]:
                continue # Ignore punctuation words

            vector = values[1:] # Don't turn into numpy array yet to avoid copying
            word_vectors.words[i] = word
            word_vectors.indices[word] = i
            word_vectors.vectors[i] = vector
            i += 1

    assert len(word_vectors.indices) == max_vocab_size + 1
    return word_vectors
