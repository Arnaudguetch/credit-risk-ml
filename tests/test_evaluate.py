

def test_predict_shape():
    import joblib
    import pandas as pd

    model = joblib.load("models/xgboost_pipeline.pkl")

    df = pd.DataFrame([{
        "age": 35,
        "sex": "male",
        "job": 2,
        "housing": "own",
        "saving accounts": "little",
        "checking account": "little",
        "credit amount": 5000,
        "duration": 24,
        "purpose": "car"
    }])

    pred = model.predict(df)

    assert len(pred) == 1