import os
import zipfile
from copy import copy
from collections import OrderedDict
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

    word_vector_size, = next(iter(word_vectors.values())).shape
    vectors = np.zeros((num_words, word_vector_size))
    # Extract word vectors from text. We start from the back to
    # get pre-padding and pre-truncation. This plays (slightly)
    # nicer with recurrent models.
    i = 0 # Negative index into vectors, in range [0, num_words)
    for word in reversed(text_to_word_sequence(text)):
        if i == num_words:
            break # Pre-truncate the rest of words (there were too many)
        try:
            vector = word_vectors[word]
        except KeyError:
            continue # Word did not have a vector. Go to the next word.
        else:
            vectors[-i] = vector
            i += 1
    if i == 0: # No word vectors found?
        raise KeyError(f"No word vectors found for {text!r}")
    else:
        return vectors

def batch(iterator, batch_size, num_words, word_vectors, dtype=np.float32):
    """Group together individual samples into batches of (features, labels)"""

    # Create fixed size lists for additional speed. Append is
    # potentially O(n) if realloc'ing, while setitem is always O(1).
    vector_size, = next(iter(word_vectors.values())).shape
    features = np.empty((batch_size, num_words, vector_size), dtype=dtype)
    labels = np.empty(batch_size, dtype=dtype)

    i = 0
    for text, label in iterator:
        # Prepare word embeddings
        try:
            vectors = prepare(text, word_vectors, num_words)
        except KeyError:
            continue # Sample did not contain any words in our vocabulary
        else:
            features[i] = vectors

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

    word_vectors = OrderedDict()
    with zipfile.ZipFile(path) as zf:
        for i, line in enumerate(zf.open(f"glove.6B.{vector_size}d.txt")):
            if i == max_vocab_size:
                break
            # Adapted from https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py#L38
            values = line.decode().split()
            word_vectors[values[0]] = np.asarray(values[1:], dtype=np.float32)

    return word_vectors
