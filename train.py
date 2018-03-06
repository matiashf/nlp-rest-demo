#!/usr/bin/env python

import os
import argparse
import math
import numpy as np
from multiprocessing import cpu_count

import keras
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import plot_model
import h5py

import data
from model import make_model, MODEL_PATH

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('--max-vocab-size', type=int, default=100000)
    parser.add_argument('--word-vector-size', type=int, default=300,
                        choices=data.word_vector_sizes)
    parser.add_argument('--num_words', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--test-fraction', type=float, default=0.25)
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-p', '--progress', action='store_true',
                        help="Show a progress bar")
    parser.add_argument('--num-samples', type=int,
                        help=("Limit the number of samples used for "
                              "training and testing"))
    args = parser.parse_args()

    print('Loading word vectors...', end=' ', flush=True)
    word_vectors = data.get_word_vectors(args.max_vocab_size, args.word_vector_size)
    print(f'{word_vectors.vectors.shape[0]} vectors loaded.')

    # Calculate test/train ratio, but allow the user to input a float
    # instead of two numbers for convenience. Scale so period ~
    # batch_size to have an acceptable rounding error.
    assert 0 < args.test_fraction < 1
    ratios = np.round(np.array([args.test_fraction, 1 - args.test_fraction])
                      * args.batch_size).astype(int)
    train_ratio, test_ratio = ratios // math.gcd(*ratios)

    # Prepare data set. This only creates *generators*, which can be
    # used for streaming and uses way less memory than loading everything at once.
    train, test = data.generators(train_ratio=test_ratio, test_ratio=test_ratio,
                                  num_words=args.num_words, word_vectors=word_vectors,
                                  batch_size=args.batch_size, seed=args.seed)

    model = make_model(num_words=args.num_words, word_vectors=word_vectors)

    model_plot_filename = os.path.join(os.path.dirname(__file__), 'model.png')
    plot_model(model, to_file=model_plot_filename)
    print(f'Saved a plot of the model structure in {model_plot_filename!r}')

    if args.num_samples is None:
        # We need to load the data set once to see how many samples have
        # enough words in the vocabulary to be included.
        print('Calculating data set size...', end=' ', flush=True)
        num_samples = data.num_samples(word_vectors)
        print(f'{num_samples} samples.')
    else:
        num_samples = args.num_samples
        print(f"Using only the first {args.num_samples} samples.")

    steps_per_epoch_factor = (num_samples
                              / ((train_ratio + test_ratio) * args.batch_size))
    history = model.fit_generator(
        train,
        steps_per_epoch=train_ratio * steps_per_epoch_factor,
        epochs=args.epochs,
        verbose=(1 if args.progress else 2),
        validation_data=test,
        validation_steps=test_ratio * steps_per_epoch_factor,
        class_weight=None, # TODO: Check for unbalanced data set
        max_queue_size=10, # Subject to tuning (CPU vs memory)
        workers=cpu_count(),
        use_multiprocessing=False, # TODO: Test speed of threads vs. processes
        shuffle=True, # TODO: Check if uses a lot of memory. If so, disable it.
        initial_epoch=0) # TODO: Allow resuming training at a later stage?

    print(f"Saving model weights to {MODEL_PATH!r}...", end=' ', flush=True)
    model.save_weights(MODEL_PATH)
    # Save max vocab size alongside the model (we need it to load word vectors)
    with h5py.File(MODEL_PATH, 'a') as f:
        f.attrs['max_vocab_size'] = args.max_vocab_size
        f.attrs['num_words'] = args.num_words
        f.attrs['word_vector_size'] = args.word_vector_size
        f.flush()
    print("done.")

if __name__ == '__main__':
    main()
