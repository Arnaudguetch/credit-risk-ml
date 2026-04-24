## 1. Projet Data Risk : Credit Risk Scoring with ML
> Credit risk scoring project with ML and MLOps.

### 1.1 Objective
> Predict customer default risk using ML models to support lending decisions.

### 1.2 Stack
- Python
- scikit-learn
- XGBoost
- MLflow
- FastAPI
- Streamlit
- Docker
- Jenkins
- Kubernetes

### 1.3 Structure
- `src/data_preprocessing` : preprocessing
- `src/train.py` : training et tracking MLflow
- `src/evaluate.py` : evaluation et graphics generation

### 1.4 Set up
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

> How to run the project :
```bash
python src/train.py
python src/evaluate.py

</> Bash
mlflow ui


