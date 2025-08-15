#!/bin/bash
# Production deployment script for pg-neo-graph-rl

set -euo pipefail

# Configuration
ENVIRONMENT="production"
REGION="us-west-2"
CLUSTER_NAME="pg-neo-graph-rl-$ENVIRONMENT"
IMAGE_TAG="${1:-latest}"

echo "🚀 Starting production deployment..."
echo "Environment: $ENVIRONMENT"
echo "Region: $REGION"
echo "Image Tag: $IMAGE_TAG"

# Check prerequisites
check_prerequisites() {
    echo "📋 Checking prerequisites..."
    
    commands=("aws" "kubectl" "helm" "terraform" "docker")
    for cmd in "${commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            echo "❌ $cmd is not installed"
            exit 1
        fi
    done
    
    echo "✅ All prerequisites met"
}

# Deploy infrastructure
deploy_infrastructure() {
    echo "🏗️ Deploying infrastructure with Terraform..."
    
    cd deployment/terraform
    
    terraform init
    terraform plan -var="image_tag=$IMAGE_TAG"
    terraform apply -var="image_tag=$IMAGE_TAG" -auto-approve
    
    cd ../..
    echo "✅ Infrastructure deployed"
}

# Update kubeconfig
update_kubeconfig() {
    echo "🔧 Updating kubeconfig..."
    aws eks update-kubeconfig --region $REGION --name $CLUSTER_NAME
    echo "✅ Kubeconfig updated"
}

# Deploy Kubernetes resources
deploy_kubernetes() {
    echo "🎯 Deploying Kubernetes resources..."
    
    # Apply in order
    kubectl apply -f deployment/kubernetes/namespace.yaml
    kubectl apply -f deployment/kubernetes/configmap.yaml
    kubectl apply -f deployment/kubernetes/secret.yaml
    kubectl apply -f deployment/kubernetes/application.yaml
    kubectl apply -f deployment/kubernetes/hpa.yaml
    kubectl apply -f deployment/kubernetes/ingress.yaml
    
    echo "✅ Kubernetes resources deployed"
}

# Wait for rollout
wait_for_rollout() {
    echo "⏳ Waiting for deployment rollout..."
    kubectl rollout status deployment/federated-graph-rl -n pg-neo-graph-rl --timeout=600s
    echo "✅ Deployment rollout complete"
}

# Run health checks
health_checks() {
    echo "🏥 Running health checks..."
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app=federated-graph-rl -n pg-neo-graph-rl --timeout=300s
    
    # Test endpoints
    SERVICE_IP=$(kubectl get svc federated-graph-rl-service -n pg-neo-graph-rl -o jsonpath='{.spec.clusterIP}')
    
    if kubectl run test-pod --image=curlimages/curl --rm -it --restart=Never -- curl -f http://$SERVICE_IP/health; then
        echo "✅ Health check passed"
    else
        echo "❌ Health check failed"
        exit 1
    fi
}

# Deploy monitoring
deploy_monitoring() {
    echo "📊 Deploying monitoring stack..."
    
    # Add Helm repositories
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    # Install Prometheus
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --values deployment/monitoring/prometheus-values.yaml
    
    # Install Grafana dashboards
    kubectl apply -f deployment/monitoring/grafana-dashboards.yaml
    
    echo "✅ Monitoring deployed"
}

# Main deployment flow
main() {
    check_prerequisites
    deploy_infrastructure
    update_kubeconfig
    deploy_kubernetes
    wait_for_rollout
    health_checks
    deploy_monitoring
    
    echo "🎉 Deployment completed successfully!"
    echo "📊 Monitor at: https://grafana.federated-rl.example.com"
    echo "🔍 Metrics at: https://prometheus.federated-rl.example.com"
    echo "🌐 Application at: https://federated-rl.example.com"
}

# Run main function
main "$@"
