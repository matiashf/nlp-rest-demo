import os

from keras.models import Sequential
from keras.layers import InputLayer, Dense, Dropout, Embedding, LSTM, Bidirectional
from keras import losses, metrics
from keras.initializers import Constant

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.h5')

def make_model(num_words, word_vectors):
    # The model below is based on
    # https://github.com/keras-team/keras/blob/master/examples/imdb_bidirectional_lstm.py

    # TODO: Control seeds used for initialization?
    model = Sequential()

    # The embedding layer looks up word vectors based on word indices
    model.add(Embedding(
        input_length=num_words, # Number of words per sample
        input_dim=len(word_vectors.words), # Size of vocabulary
        output_dim=word_vectors.vectors.shape[-1], # Word vector size
        embeddings_initializer=Constant(word_vectors.vectors),
        embeddings_regularizer=None,
        activity_regularizer=None,
        embeddings_constraint=None,
        mask_zero=True,
        trainable=False))

    # LSTM
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))

    # Final dense layer for binary classification
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss=losses.binary_crossentropy,
                  metrics=[metrics.binary_accuracy])

    return model
