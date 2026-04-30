import os


def test_model_exists():
    assert os.path.exists("models/xgboost_pipeline.pkl")
    