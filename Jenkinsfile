pipeline {
    agent any

    environment {
        IMAGE_NAME = "credit-risk-api"
        IMAGE_TAG = "${BUILD_NUMBER}"
        DOCKER_REGISTRY = "docker.io"
        DOCKER_REPO = "/credit-risk-ml"
        FULL_IMAGE = "${DOCKER_REGISTRY}/${DOCKER_REPO}:${IMAGE_TAG}"
        KUBE_NAMESPACE = "credit-risk"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        stage('Create virtual environment and install dependencies') {
            steps {
                sh '''
                python3 -m venv venv
                . venv/bin/activate
                python -m pip install --upgrade pip
                pip install -r requirements.txt
                '''
            }
        }
        stage('Run tests') {
            steps {
                sh '''
                . venv/bin/activate
                pytest tests/
                '''
            }
        }
        stage('Train model') {
            steps {
                sh '''
                . venv/bin/activate
                python src/train.py
                '''
            }
        }
        stage('Evaluate model') {
            steps {
                sh '''
                . venv/bin/activate
                python src/evaluate.py
                '''
            }
        }
        stage('SHAP report') {
            steps {
                sh '''
                . venv/bin/activate
                python src/explain.py
                '''
            }
        }
        stage('Archive reports and model') {
            steps {
                archiveArtifacts artifacts: 'models/*pkl,reports/*png,reports/*csv', fingerprint: true          
            }
        }
        stage("Build Docker image") {
            steps {
                sh '''
                docker build -t ${FULL_IMAGE} .
                docker tag ${FULL_IMAGE} ${DOCKER_REGISTRY}/${DOCKER_REPO}:latest
                '''
            }
        }
        stage('Push Docker image') {
            steps {
                withCredentials([usernamePassword(
                    credentialID: 'docker-creds',
                    usernameVariablle: 'DOCKER_USER',
                    passwordVariable: 'DOCKER_PASS'
                )]) {
                    sh '''
                    echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" -password-stdin
                    docker push ${FULL_IMAGE}
                    docker push ${DOCKER_REGISTRY}/${DOCKER_REPO}:latest
                    '''
                }
            }
        }
        stage('Deploy to kubernetes') {
            steps {
                withCredentials([file(
                    credentialsID: 'kubeconfig',
                    variable: 'KUBECONFIG_FILE'
                )]) {
                    sh '''
                    export KUBECONFIG=$KUBECONFIG_FILE
                    kubectl create namespace ${KUBE_NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

                    sed "s|IMAGE_PLACEHOLDER|${FULL_IMAGE}|g" k8s/deployment.yaml | kubectl apply -n ${KUBE_NAMESPACE} -f -
                    kubectl apply -n ${KUBE_NAMESPACE} -f k8s/service.yaml

                    kubectl rollout status deployment/credit-risk-api -n ${KUBE_NAMESPACE}
                    '''
                }
            }
        }
    }

    post {
        success {
            echo 'CI/CD pipeline completed successfully'       
        }
        failure {
            echo 'CI/CD pipeline failed'
        }
    }
}