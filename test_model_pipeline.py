"""Tests unitaires pour model_pipeline.py."""

from model_pipeline import prepare_data, train_model


def test_prepare_data_shapes():
    """Vérifie les dimensions des données préparées."""
    x_train, x_test, y_train, y_test = prepare_data()
    assert x_train.shape[0] == 8000
    assert x_test.shape[0] == 2000
    assert x_train.shape[1] == 9
    assert len(y_train) == 8000


def test_train_model_predicts():
    """Vérifie que le modèle entraîné produit des prédictions 0/1."""
    x_train, x_test, y_train, y_test = prepare_data()
    model = train_model(x_train, y_train, n_estimators=10)
    preds = model.predict(x_test)
    assert len(preds) == len(y_test)
    assert set(preds) <= {0, 1}
