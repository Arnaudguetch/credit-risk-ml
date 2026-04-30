

def test_predict_shape():
    import joblib
    import pandas as pd

    model = joblib.load("models/xgboost_pipeline.pkl")

    df = pd.DataFrame([{
    "Age": 35,
    "Sex": "male",
    "Job": 2,
    "Housing": "own",
    "Saving accounts": "little",
    "Checking account": "little",
    "Credit amount": 5000,
    "Duration": 24,
    "Purpose": "car"
}])

    pred = model.predict(df)

    assert len(pred) == 1