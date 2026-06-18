#!/bin/bash

echo "Stopping applications..."

kubectl scale deployment credit-risk-api --replicas=0
kubectl scale deployment credit-risk-streamlit --replicas=0
kubectl scale deployment mlflow --replicas=0
kubectl scale deployment prometheus-server --replicas=0 -n monitoring
kubectl scale deployment grafana --replicas=0 -n monitoring

echo "Done!"
kubectl get pods
kubectl get deployments -A