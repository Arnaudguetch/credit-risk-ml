#!/bin/bash

echo "Starting applications..."

kubectl scale deployment credit-risk-api --replicas=1
kubectl scale deployment credit-risk-streamlit --replicas=1
kubectl scale deployment mlflow --replicas=1
kubectl scale deployment prometheus-server --replicas=1 -n monitoring
kubectl scale deployment grafana --replicas=1 -n monitoring

echo "Done!"
kubectl get pods
kubectl get deployments -A