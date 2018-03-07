import os

import keras
import numpy as np
import h5py
import flask

import data
from model import MODEL_PATH, make_model

assert os.path.exists(MODEL_PATH)

print('Loading model...')
with h5py.File(MODEL_PATH, 'a') as f:
    max_vocab_size = f.attrs['max_vocab_size']
    num_words = f.attrs['num_words']
    word_vector_size = f.attrs['word_vector_size']
word_vectors = data.get_word_vectors(max_vocab_size, word_vector_size)
model = make_model(num_words=num_words, word_vectors=word_vectors)
model.load_weights(MODEL_PATH)
print('Model loaded.')

app = flask.Flask(__name__)

@app.route("/", methods=["GET"])
def main():
    return flask.render_template('main.html')

# Work around bug in Keras + TensorFlow when using multiple
# threads. Idea from
# https://github.com/keras-team/keras/issues/2397#issuecomment-254919212
assert keras.backend.backend() == 'tensorflow'
import tensorflow as tf
global_graph = tf.get_default_graph()

@app.route("/predict", methods=["GET"])
def predict():
    global global_graph
    # Based on
    # https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

    # TODO: Handle errors gracefully
    text = flask.request.args.get('text', type=str)

    if text is None:
        error = "Missing required parameter 'text'"
        success = False
    else:
        try:
            features = data.prepare(text, word_vectors, num_words)
        except data.InvalidTextError as e:
            error = str(e)
            success = False
        else:
            # Turn into a batch of one for prediction
            with global_graph.as_default():
                prediction = model.predict(features[np.newaxis, :], batch_size=1)
            sentiment = ('negative', 'positive')[int(prediction.round())]
            positivity_score = float(prediction)
            words = word_vectors.words[features[features != 0]].tolist()
            success = True

    allowed = set('text error success sentiment positivity_score words'.split())
    return flask.jsonify({k: v for k, v in locals().items() if k in allowed})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
