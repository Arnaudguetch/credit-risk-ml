## 1. Projet Data Risk : Credit Risk Scoring with ML
> Credit risk scoring project with ML and MLOps.

### 1.1 Objective
> Predict customer default risk using ML models to support lending decisions.

Credit-Risk-Scoring
  |
  |-api
  |  
  |-data
  |  |-processed
  |  |-raw
  |   
  |-models
  |
  |-notebooks
  |
  |-reports
  |
  |-src
  |
  |-k8s
  |
  |-.github
  |   |workflow

  

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
python src/data_preprocessing.py
python src/train.py
python src/evaluate.py
python src/explain.py

</> Bash
mlflow ui -> running mlflow app
uvicorn api.app:app --reload  && then http://127.0.0.1:8000/docs -> running the serving app

streamlit run streamlit_app.py -> to run streamlit app

docker build -t credit-risk-api . -> to build docker image
docker run -p 8000:8000 credit-risk-api -> to run docker image

http://localhost:8000/docs -> to show the Swagger doc 

### 1.4 CI/CD

This projet uuses Github Actions to:
- run tests
- train and evaluate the model
- generate SHAP reports
- build the Docker image
- deploy the API to a temporary Minikube Kuberntes cluster on Azure
- validate the '/health' endpoint

This avoids clouds costs while demonstrating a complete kubernetes CI/CD worflow.

