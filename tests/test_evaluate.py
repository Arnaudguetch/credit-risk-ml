

def test_predict_shape():
    import joblib
    import pandas as pd

    model = joblib.load("models/xgboost_pipeline.pkl")

    df = pd.DataFrame([{
        "age": 35,
        "credit_amount": 5000,
        "duration": 24
    }])

    pred = model.predict(df)

    assert len(pred) == 1