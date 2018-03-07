import pytest

@pytest.fixture(scope='session')
def app():
    from web import app
    app.testing = True
    return app

@pytest.fixture
def client(app):
    with app.test_client() as client:
        yield client

def test_main(client):
    client.get('/')

def test_predict(client):
    client.get('/predict')

    client.get('/predict?text=test')
