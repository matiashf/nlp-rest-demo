# Demo of Natural Language Processing + REST (web services) in Python

# Setup

This project is best used with [pyenv](https://github.com/pyenv/pyenv)
and [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) to
create a local development environment:

```sh
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc
exec "$SHELL" # Restart your shell to load the new configuration

git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
exec "$SHELL"
```

After setting up pyenv you can install build requirements for python,
build the correct version and install necessary python libraries.

```sh
cat packages.txt | xargs sudo apt-get install -y 
pyenv install $(cat .python-version)
pip install -r requirements.txt
```

# Data set for training

Go to the [the Yelp dataset on Kaggle](https://www.kaggle.com/yelp-dataset/yelp-dataset/data), log in and accept the usage agreement for the project,
then download [`yelp_review.csv`](https://www.kaggle.com/yelp-dataset/yelp-dataset/downloads/yelp_review.csv). The file should be stored in the repository root folder, and doesn't need to be unzipped.

# References

Words are represented using word vectors from the [GloVe (Global Vectors for Word
Representation)](https://nlp.stanford.edu/projects/glove/) project.
