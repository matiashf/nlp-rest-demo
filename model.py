import os

from keras.models import Sequential
from keras.layers import InputLayer, Dense, Dropout, Embedding, LSTM, Bidirectional

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.h5')

def make_model(num_words, word_vector_size):
    # The model below is based on
    # https://github.com/keras-team/keras/blob/master/examples/imdb_bidirectional_lstm.py

    # TODO: Control seeds used for initialization?
    model = Sequential()
    model.add(InputLayer(input_shape=(num_words, word_vector_size), dtype='float32'))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model
