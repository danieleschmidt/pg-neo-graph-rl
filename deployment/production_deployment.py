"""Production deployment orchestrator for pg-neo-graph-rl."""
import os
import time
import json
import subprocess
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DeploymentConfig:
    """Production deployment configuration."""
    # Infrastructure
    environment: str = "production"
    region: str = "us-west-2"
    availability_zones: List[str] = None
    
    # Scaling
    min_instances: int = 2
    max_instances: int = 100
    target_cpu_utilization: int = 70
    
    # Networking
    enable_load_balancer: bool = True
    ssl_certificate_arn: str = ""
    domain_name: str = ""
    
    # Storage
    database_instance_class: str = "db.r5.large"
    enable_encryption: bool = True
    backup_retention_days: int = 30
    
    # Monitoring
    enable_cloudwatch: bool = True
    enable_prometheus: bool = True
    enable_grafana: bool = True
    log_retention_days: int = 30
    
    # Security
    enable_vpc: bool = True
    enable_waf: bool = True
    enable_secrets_manager: bool = True
    
    # Performance
    enable_caching: bool = True
    cache_instance_type: str = "cache.r6g.large"
    
    def __post_init__(self):
        if self.availability_zones is None:
            self.availability_zones = [f"{self.region}a", f"{self.region}b", f"{self.region}c"]

class ProductionDeployer:
    """Orchestrates production deployment of federated learning system."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.deployment_dir = Path("deployment")
        self.deployment_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.deployment_dir / "terraform").mkdir(exist_ok=True)
        (self.deployment_dir / "kubernetes").mkdir(exist_ok=True)
        (self.deployment_dir / "docker").mkdir(exist_ok=True)
        (self.deployment_dir / "monitoring").mkdir(exist_ok=True)
        (self.deployment_dir / "scripts").mkdir(exist_ok=True)
        
        logger.info(f"Production deployer initialized for {config.environment}")
    
    def generate_deployment_artifacts(self) -> None:
        """Generate all deployment artifacts."""
        logger.info("Generating deployment artifacts...")
        
        # Generate infrastructure as code
        self._generate_terraform_config()
        
        # Generate Kubernetes manifests
        self._generate_kubernetes_manifests()
        
        # Generate Docker configurations
        self._generate_docker_configs()
        
        # Generate monitoring configurations
        self._generate_monitoring_configs()
        
        # Generate deployment scripts
        self._generate_deployment_scripts()
        
        # Generate CI/CD pipeline
        self._generate_cicd_pipeline()
        
        logger.info("✅ All deployment artifacts generated")
    
    def _generate_terraform_config(self) -> None:
        """Generate Terraform infrastructure configuration."""
        terraform_dir = self.deployment_dir / "terraform"
        
        # Main Terraform configuration
        main_tf = f"""# Production Infrastructure for pg-neo-graph-rl
terraform {{
  required_version = ">= 1.0"
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }}
    kubernetes = {{
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }}
    helm = {{
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }}
  }}
  
  backend "s3" {{
    bucket = "pg-neo-graph-rl-terraform-state"
    key    = "{self.config.environment}/terraform.tfstate"
    region = "{self.config.region}"
    encrypt = true
    dynamodb_table = "terraform-state-lock"
  }}
}}

provider "aws" {{
  region = "{self.config.region}"
  
  default_tags {{
    tags = {{
      Environment = "{self.config.environment}"
      Project     = "pg-neo-graph-rl"
      ManagedBy   = "terraform"
    }}
  }}
}}

# Data sources
data "aws_availability_zones" "available" {{
  state = "available"
}}

# VPC and Networking
module "vpc" {{
  source = "./modules/vpc"
  
  environment = "{self.config.environment}"
  cidr_block = "10.0.0.0/16"
  availability_zones = {json.dumps(self.config.availability_zones)}
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
  enable_dns_hostnames = true
  enable_dns_support = true
  
  tags = {{
    Name = "pg-neo-graph-rl-vpc"
  }}
}}

# EKS Cluster
module "eks" {{
  source = "./modules/eks"
  
  cluster_name = "pg-neo-graph-rl-{self.config.environment}"
  cluster_version = "1.27"
  
  vpc_id = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  node_groups = {{
    general = {{
      instance_types = ["m5.large", "m5.xlarge"]
      capacity_type = "ON_DEMAND"
      min_size = {self.config.min_instances}
      max_size = {self.config.max_instances}
      desired_size = {self.config.min_instances + 1}
    }}
    compute = {{
      instance_types = ["c5.2xlarge", "c5.4xlarge"]
      capacity_type = "SPOT"
      min_size = 0
      max_size = 50
      desired_size = 5
    }}
    gpu = {{
      instance_types = ["p3.2xlarge", "g4dn.xlarge"]
      capacity_type = "ON_DEMAND"
      min_size = 0
      max_size = 10
      desired_size = 0
    }}
  }}
  
  enable_irsa = true
}}

# RDS Database
module "database" {{
  source = "./modules/rds"
  
  identifier = "pg-neo-graph-rl-{self.config.environment}"
  engine = "postgres"
  engine_version = "14.9"
  instance_class = "{self.config.database_instance_class}"
  
  allocated_storage = 100
  max_allocated_storage = 1000
  storage_encrypted = {str(self.config.enable_encryption).lower()}
  
  vpc_id = module.vpc.vpc_id
  subnet_ids = module.vpc.database_subnets
  
  backup_retention_period = {self.config.backup_retention_days}
  backup_window = "03:00-04:00"
  maintenance_window = "sun:04:00-sun:05:00"
}}

# ElastiCache Redis
module "cache" {{
  source = "./modules/elasticache"
  
  cluster_id = "pg-neo-graph-rl-{self.config.environment}"
  node_type = "{self.config.cache_instance_type}"
  num_cache_nodes = 3
  
  vpc_id = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  at_rest_encryption_enabled = {str(self.config.enable_encryption).lower()}
  transit_encryption_enabled = {str(self.config.enable_encryption).lower()}
}}

# Application Load Balancer
module "alb" {{
  source = "./modules/alb"
  count = {str(self.config.enable_load_balancer).lower()} ? 1 : 0
  
  name = "pg-neo-graph-rl-{self.config.environment}"
  vpc_id = module.vpc.vpc_id
  subnet_ids = module.vpc.public_subnets
  
  certificate_arn = "{self.config.ssl_certificate_arn}"
  domain_name = "{self.config.domain_name}"
}}

# Monitoring and Observability
module "monitoring" {{
  source = "./modules/monitoring"
  
  cluster_name = module.eks.cluster_name
  vpc_id = module.vpc.vpc_id
  
  enable_prometheus = {str(self.config.enable_prometheus).lower()}
  enable_grafana = {str(self.config.enable_grafana).lower()}
  enable_cloudwatch = {str(self.config.enable_cloudwatch).lower()}
  
  log_retention_days = {self.config.log_retention_days}
}}

# Security
module "security" {{
  source = "./modules/security"
  
  vpc_id = module.vpc.vpc_id
  cluster_name = module.eks.cluster_name
  
  enable_waf = {str(self.config.enable_waf).lower()}
  enable_secrets_manager = {str(self.config.enable_secrets_manager).lower()}
}}

# Outputs
output "cluster_endpoint" {{
  description = "EKS cluster endpoint"
  value = module.eks.cluster_endpoint
}}

output "database_endpoint" {{
  description = "RDS database endpoint"
  value = module.database.endpoint
}}

output "redis_endpoint" {{
  description = "ElastiCache Redis endpoint"
  value = module.cache.endpoint
}}

output "load_balancer_dns" {{
  description = "Application Load Balancer DNS"
  value = {str(self.config.enable_load_balancer).lower()} ? module.alb[0].dns_name : null
}}
"""
        
        with open(terraform_dir / "main.tf", "w") as f:
            f.write(main_tf)
        
        # Variables file
        variables_tf = f"""# Terraform variables for pg-neo-graph-rl
variable "environment" {{
  description = "Deployment environment"
  type = string
  default = "{self.config.environment}"
}}

variable "region" {{
  description = "AWS region"
  type = string
  default = "{self.config.region}"
}}

variable "min_instances" {{
  description = "Minimum number of instances"
  type = number
  default = {self.config.min_instances}
}}

variable "max_instances" {{
  description = "Maximum number of instances"
  type = number
  default = {self.config.max_instances}
}}

variable "database_instance_class" {{
  description = "RDS instance class"
  type = string
  default = "{self.config.database_instance_class}"
}}

variable "enable_encryption" {{
  description = "Enable encryption at rest"
  type = bool
  default = {str(self.config.enable_encryption).lower()}
}}
"""
        
        with open(terraform_dir / "variables.tf", "w") as f:
            f.write(variables_tf)
        
        logger.info("✅ Terraform configuration generated")
    
    def _generate_kubernetes_manifests(self) -> None:
        """Generate Kubernetes deployment manifests."""
        k8s_dir = self.deployment_dir / "kubernetes"
        
        # Namespace
        namespace_yaml = f"""apiVersion: v1
kind: Namespace
metadata:
  name: pg-neo-graph-rl
  labels:
    name: pg-neo-graph-rl
    environment: {self.config.environment}
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: compute-quota
  namespace: pg-neo-graph-rl
spec:
  hard:
    requests.cpu: "100"
    requests.memory: 200Gi
    limits.cpu: "200"
    limits.memory: 400Gi
    persistentvolumeclaims: "10"
"""
        
        with open(k8s_dir / "namespace.yaml", "w") as f:
            f.write(namespace_yaml)
        
        # Core application deployment
        app_deployment = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: federated-graph-rl
  namespace: pg-neo-graph-rl
  labels:
    app: federated-graph-rl
    version: v1
spec:
  replicas: {self.config.min_instances}
  selector:
    matchLabels:
      app: federated-graph-rl
  template:
    metadata:
      labels:
        app: federated-graph-rl
        version: v1
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
        fsGroup: 10001
      containers:
      - name: federated-graph-rl
        image: pg-neo-graph-rl:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP
        - containerPort: 8001
          name: metrics
          protocol: TCP
        env:
        - name: ENVIRONMENT
          value: "{self.config.environment}"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: cache-credentials
              key: url
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: cache
          mountPath: /app/cache
      volumes:
      - name: tmp
        emptyDir: {{}}
      - name: cache
        emptyDir: {{}}
---
apiVersion: v1
kind: Service
metadata:
  name: federated-graph-rl-service
  namespace: pg-neo-graph-rl
  labels:
    app: federated-graph-rl
spec:
  selector:
    app: federated-graph-rl
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  - name: metrics
    port: 8001
    targetPort: 8001
    protocol: TCP
  type: ClusterIP
"""
        
        with open(k8s_dir / "application.yaml", "w") as f:
            f.write(app_deployment)
        
        # Horizontal Pod Autoscaler
        hpa_yaml = f"""apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: federated-graph-rl-hpa
  namespace: pg-neo-graph-rl
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: federated-graph-rl
  minReplicas: {self.config.min_instances}
  maxReplicas: {self.config.max_instances}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: {self.config.target_cpu_utilization}
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
"""
        
        with open(k8s_dir / "hpa.yaml", "w") as f:
            f.write(hpa_yaml)
        
        logger.info("✅ Kubernetes manifests generated")
    
    def _generate_docker_configs(self) -> None:
        """Generate Docker configurations."""
        docker_dir = self.deployment_dir / "docker"
        
        # Production Dockerfile
        dockerfile = """# Production Dockerfile for pg-neo-graph-rl
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1 \\
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    git \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser -u 10001 appuser

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt requirements-automation.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \\
    pip install --no-cache-dir -r requirements-automation.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Install application
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /app/data && \\
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "pg_neo_graph_rl.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Multi-stage build for GPU support
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu

ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1 \\
    DEBIAN_FRONTEND=noninteractive

# Install Python 3.11
RUN apt-get update && apt-get install -y \\
    software-properties-common \\
    && add-apt-repository ppa:deadsnakes/ppa \\
    && apt-get update && apt-get install -y \\
    python3.11 \\
    python3.11-pip \\
    python3.11-dev \\
    build-essential \\
    curl \\
    git \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

# Create symlinks
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \\
    ln -s /usr/bin/python3.11 /usr/bin/python3

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser -u 10001 appuser

# Set work directory
WORKDIR /app

# Install Python dependencies with GPU support
COPY requirements.txt requirements-automation.txt ./
RUN python -m pip install --upgrade pip && \\
    python -m pip install --no-cache-dir -r requirements.txt && \\
    python -m pip install --no-cache-dir -r requirements-automation.txt && \\
    python -m pip install --no-cache-dir "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Copy application code
COPY --chown=appuser:appuser . .

# Install application
RUN python -m pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /app/data && \\
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "pg_neo_graph_rl.api:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        with open(docker_dir / "Dockerfile", "w") as f:
            f.write(dockerfile)
        
        # Docker Compose for local development and testing
        docker_compose = f"""version: '3.8'

services:
  app:
    build:
      context: ../..
      dockerfile: deployment/docker/Dockerfile
      target: base
    ports:
      - "8000:8000"
      - "8001:8001"
    environment:
      - ENVIRONMENT=development
      - DATABASE_URL=postgresql://postgres:password@db:5432/pgneographrl
      - REDIS_URL=redis://redis:6379/0
      - LOG_LEVEL=DEBUG
    volumes:
      - ../../logs:/app/logs
      - ../../data:/app/data
    depends_on:
      - db
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:14-alpine
    environment:
      - POSTGRES_DB=pgneographrl
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    name: pg-neo-graph-rl-network
"""
        
        with open(docker_dir / "docker-compose.yml", "w") as f:
            f.write(docker_compose)
        
        logger.info("✅ Docker configurations generated")
    
    def _generate_monitoring_configs(self) -> None:
        """Generate monitoring configurations."""
        monitoring_dir = self.deployment_dir / "monitoring"
        
        # Prometheus configuration
        prometheus_config = """global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'federated-graph-rl'
    static_configs:
      - targets: ['app:8001']
    metrics_path: /metrics
    scrape_interval: 10s
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
      
  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']
      
  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - pg-neo-graph-rl
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\\d+)?;(\\d+)
        replacement: $1:$2
        target_label: __address__
"""
        
        with open(monitoring_dir / "prometheus.yml", "w") as f:
            f.write(prometheus_config)
        
        # Alert rules
        alert_rules = """groups:
- name: federated_graph_rl_alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"
      
  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High latency detected"
      description: "95th percentile latency is {{ $value }} seconds"
      
  - alert: MemoryUsageHigh
    expr: (container_memory_usage_bytes / container_spec_memory_limit_bytes) > 0.85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage"
      description: "Memory usage is {{ $value | humanizePercentage }}"
      
  - alert: CPUUsageHigh
    expr: rate(container_cpu_usage_seconds_total[5m]) > 0.8
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage"
      description: "CPU usage is {{ $value | humanizePercentage }}"
      
  - alert: FederatedLearningConvergenceIssue
    expr: increase(federated_learning_convergence_failures_total[1h]) > 5
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Federated learning convergence issues"
      description: "{{ $value }} convergence failures in the last hour"
"""
        
        with open(monitoring_dir / "alert_rules.yml", "w") as f:
            f.write(alert_rules)
        
        logger.info("✅ Monitoring configurations generated")
    
    def _generate_deployment_scripts(self) -> None:
        """Generate deployment scripts."""
        scripts_dir = self.deployment_dir / "scripts"
        
        # Main deployment script
        deploy_script = f"""#!/bin/bash
# Production deployment script for pg-neo-graph-rl

set -euo pipefail

# Configuration
ENVIRONMENT="{self.config.environment}"
REGION="{self.config.region}"
CLUSTER_NAME="pg-neo-graph-rl-$ENVIRONMENT"
IMAGE_TAG="${{1:-latest}}"

echo "🚀 Starting production deployment..."
echo "Environment: $ENVIRONMENT"
echo "Region: $REGION"
echo "Image Tag: $IMAGE_TAG"

# Check prerequisites
check_prerequisites() {{
    echo "📋 Checking prerequisites..."
    
    commands=("aws" "kubectl" "helm" "terraform" "docker")
    for cmd in "${{commands[@]}}"; do
        if ! command -v "$cmd" &> /dev/null; then
            echo "❌ $cmd is not installed"
            exit 1
        fi
    done
    
    echo "✅ All prerequisites met"
}}

# Deploy infrastructure
deploy_infrastructure() {{
    echo "🏗️ Deploying infrastructure with Terraform..."
    
    cd deployment/terraform
    
    terraform init
    terraform plan -var="image_tag=$IMAGE_TAG"
    terraform apply -var="image_tag=$IMAGE_TAG" -auto-approve
    
    cd ../..
    echo "✅ Infrastructure deployed"
}}

# Update kubeconfig
update_kubeconfig() {{
    echo "🔧 Updating kubeconfig..."
    aws eks update-kubeconfig --region $REGION --name $CLUSTER_NAME
    echo "✅ Kubeconfig updated"
}}

# Deploy Kubernetes resources
deploy_kubernetes() {{
    echo "🎯 Deploying Kubernetes resources..."
    
    # Apply in order
    kubectl apply -f deployment/kubernetes/namespace.yaml
    kubectl apply -f deployment/kubernetes/configmap.yaml
    kubectl apply -f deployment/kubernetes/secret.yaml
    kubectl apply -f deployment/kubernetes/application.yaml
    kubectl apply -f deployment/kubernetes/hpa.yaml
    kubectl apply -f deployment/kubernetes/ingress.yaml
    
    echo "✅ Kubernetes resources deployed"
}}

# Wait for rollout
wait_for_rollout() {{
    echo "⏳ Waiting for deployment rollout..."
    kubectl rollout status deployment/federated-graph-rl -n pg-neo-graph-rl --timeout=600s
    echo "✅ Deployment rollout complete"
}}

# Run health checks
health_checks() {{
    echo "🏥 Running health checks..."
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app=federated-graph-rl -n pg-neo-graph-rl --timeout=300s
    
    # Test endpoints
    SERVICE_IP=$(kubectl get svc federated-graph-rl-service -n pg-neo-graph-rl -o jsonpath='{{.spec.clusterIP}}')
    
    if kubectl run test-pod --image=curlimages/curl --rm -it --restart=Never -- curl -f http://$SERVICE_IP/health; then
        echo "✅ Health check passed"
    else
        echo "❌ Health check failed"
        exit 1
    fi
}}

# Deploy monitoring
deploy_monitoring() {{
    echo "📊 Deploying monitoring stack..."
    
    # Add Helm repositories
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    # Install Prometheus
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \\
        --namespace monitoring \\
        --create-namespace \\
        --values deployment/monitoring/prometheus-values.yaml
    
    # Install Grafana dashboards
    kubectl apply -f deployment/monitoring/grafana-dashboards.yaml
    
    echo "✅ Monitoring deployed"
}}

# Main deployment flow
main() {{
    check_prerequisites
    deploy_infrastructure
    update_kubeconfig
    deploy_kubernetes
    wait_for_rollout
    health_checks
    deploy_monitoring
    
    echo "🎉 Deployment completed successfully!"
    echo "📊 Monitor at: https://grafana.{self.config.domain_name}"
    echo "🔍 Metrics at: https://prometheus.{self.config.domain_name}"
    echo "🌐 Application at: https://{self.config.domain_name}"
}}

# Run main function
main "$@"
"""
        
        with open(scripts_dir / "deploy.sh", "w") as f:
            f.write(deploy_script)
        
        # Make script executable
        os.chmod(scripts_dir / "deploy.sh", 0o755)
        
        # Rollback script
        rollback_script = """#!/bin/bash
# Rollback script for pg-neo-graph-rl

set -euo pipefail

ENVIRONMENT="${ENVIRONMENT:-production}"
PREVIOUS_VERSION="${1:-}"

if [ -z "$PREVIOUS_VERSION" ]; then
    echo "Usage: $0 <previous_version>"
    echo "Example: $0 v1.2.3"
    exit 1
fi

echo "🔄 Rolling back to version: $PREVIOUS_VERSION"

# Rollback Kubernetes deployment
kubectl rollout undo deployment/federated-graph-rl -n pg-neo-graph-rl

# Wait for rollback to complete
kubectl rollout status deployment/federated-graph-rl -n pg-neo-graph-rl --timeout=600s

# Verify rollback
kubectl wait --for=condition=ready pod -l app=federated-graph-rl -n pg-neo-graph-rl --timeout=300s

echo "✅ Rollback completed successfully"
"""
        
        with open(scripts_dir / "rollback.sh", "w") as f:
            f.write(rollback_script)
        
        os.chmod(scripts_dir / "rollback.sh", 0o755)
        
        logger.info("✅ Deployment scripts generated")
    
    def _generate_cicd_pipeline(self) -> None:
        """Generate CI/CD pipeline configuration."""
        # GitHub Actions workflow
        github_workflow = f"""name: Production Deployment

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{{{ github.repository }}}}
  AWS_REGION: {self.config.region}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .
        
    - name: Run tests
      run: |
        python -m pytest tests/ -v --tb=short
        
    - name: Run security scan
      run: |
        pip install bandit safety
        bandit -r pg_neo_graph_rl/
        safety check
        
    - name: Run linting
      run: |
        pip install ruff black mypy
        ruff check pg_neo_graph_rl/
        black --check pg_neo_graph_rl/
        mypy pg_neo_graph_rl/

  build:
    needs: test
    runs-on: ubuntu-latest
    outputs:
      image: ${{{{ steps.image.outputs.image }}}}
      digest: ${{{{ steps.build.outputs.digest }}}}
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
      
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{{{ env.REGISTRY }}}}
        username: ${{{{ github.actor }}}}
        password: ${{{{ secrets.GITHUB_TOKEN }}}}
        
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{{{ env.REGISTRY }}}}/${{{{ env.IMAGE_NAME }}}}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{{{version}}}}
          type=semver,pattern={{{{major}}}}.{{{{minor}}}}
          type=sha,prefix={{{{branch}}}}-
          
    - name: Build and push Docker image
      id: build
      uses: docker/build-push-action@v5
      with:
        context: .
        file: deployment/docker/Dockerfile
        push: true
        tags: ${{{{ steps.meta.outputs.tags }}}}
        labels: ${{{{ steps.meta.outputs.labels }}}}
        cache-from: type=gha
        cache-to: type=gha,mode=max
        
    - name: Output image
      id: image
      run: |
        echo "image=${{{{ env.REGISTRY }}}}/${{{{ env.IMAGE_NAME }}}}" >> $GITHUB_OUTPUT

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' || startsWith(github.ref, 'refs/tags/v')
    environment: {self.config.environment}
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{{{ secrets.AWS_ACCESS_KEY_ID }}}}
        aws-secret-access-key: ${{{{ secrets.AWS_SECRET_ACCESS_KEY }}}}
        aws-region: ${{{{ env.AWS_REGION }}}}
        
    - name: Setup Terraform
      uses: hashicorp/setup-terraform@v3
      with:
        terraform_version: 1.5.0
        
    - name: Setup kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'v1.27.0'
        
    - name: Setup Helm
      uses: azure/setup-helm@v3
      with:
        version: '3.12.0'
        
    - name: Deploy infrastructure
      run: |
        cd deployment/terraform
        terraform init
        terraform plan -var="image_tag=${{{{ github.sha }}}}"
        terraform apply -var="image_tag=${{{{ github.sha }}}}" -auto-approve
        
    - name: Update kubeconfig
      run: |
        aws eks update-kubeconfig --region ${{{{ env.AWS_REGION }}}} --name pg-neo-graph-rl-{self.config.environment}
        
    - name: Deploy to Kubernetes
      run: |
        # Update image in deployment
        sed -i 's|image: pg-neo-graph-rl:latest|image: ${{{{ needs.build.outputs.image }}}}@${{{{ needs.build.outputs.digest }}}}|g' deployment/kubernetes/application.yaml
        
        kubectl apply -f deployment/kubernetes/
        kubectl rollout status deployment/federated-graph-rl -n pg-neo-graph-rl --timeout=600s
        
    - name: Run smoke tests
      run: |
        kubectl wait --for=condition=ready pod -l app=federated-graph-rl -n pg-neo-graph-rl --timeout=300s
        
        # Test health endpoint
        SERVICE_IP=$(kubectl get svc federated-graph-rl-service -n pg-neo-graph-rl -o jsonpath='{{.spec.clusterIP}}')
        kubectl run smoke-test --image=curlimages/curl --rm -it --restart=Never -- curl -f http://$SERVICE_IP/health
        
    - name: Notify deployment success
      if: success()
      run: |
        echo "🎉 Deployment to {self.config.environment} completed successfully!"
        echo "Image: ${{{{ needs.build.outputs.image }}}}@${{{{ needs.build.outputs.digest }}}}"
"""
        
        # Create .github/workflows directory
        github_dir = Path(".github/workflows")
        github_dir.mkdir(parents=True, exist_ok=True)
        
        with open(github_dir / "production-deployment.yml", "w") as f:
            f.write(github_workflow)
        
        logger.info("✅ CI/CD pipeline generated")
    
    def generate_deployment_summary(self) -> Dict[str, Any]:
        """Generate deployment summary and checklist."""
        summary = {
            "environment": self.config.environment,
            "region": self.config.region,
            "infrastructure": {
                "min_instances": self.config.min_instances,
                "max_instances": self.config.max_instances,
                "database_class": self.config.database_instance_class,
                "cache_type": self.config.cache_instance_type,
                "encryption_enabled": self.config.enable_encryption
            },
            "features": {
                "auto_scaling": True,
                "load_balancing": self.config.enable_load_balancer,
                "monitoring": self.config.enable_prometheus and self.config.enable_grafana,
                "security": self.config.enable_waf and self.config.enable_secrets_manager,
                "ssl": bool(self.config.ssl_certificate_arn)
            },
            "deployment_artifacts": [
                "Terraform infrastructure code",
                "Kubernetes manifests",
                "Docker configurations", 
                "Monitoring setup",
                "Deployment scripts",
                "CI/CD pipeline"
            ],
            "pre_deployment_checklist": [
                "Configure AWS credentials and permissions",
                "Set up domain name and SSL certificate",
                "Create S3 bucket for Terraform state",
                "Configure secrets in AWS Secrets Manager",
                "Set up monitoring and alerting channels",
                "Review security configurations",
                "Test deployment in staging environment"
            ],
            "post_deployment_checklist": [
                "Verify all services are healthy",
                "Test federated learning functionality",
                "Validate monitoring and alerting",
                "Run performance benchmarks",
                "Document operational procedures",
                "Set up backup and disaster recovery",
                "Train operations team"
            ]
        }
        
        return summary

if __name__ == "__main__":
    # Example usage
    config = DeploymentConfig(
        environment="production",
        region="us-west-2",
        min_instances=3,
        max_instances=50,
        domain_name="federated-rl.example.com",
        ssl_certificate_arn="arn:aws:acm:us-west-2:123456789012:certificate/12345678-1234-1234-1234-123456789012"
    )
    
    deployer = ProductionDeployer(config)
    deployer.generate_deployment_artifacts()
    
    summary = deployer.generate_deployment_summary()
    
    print("🚀 Production Deployment Artifacts Generated!")
    print(f"Environment: {summary['environment']}")
    print(f"Region: {summary['region']}")
    print(f"Instance Range: {summary['infrastructure']['min_instances']}-{summary['infrastructure']['max_instances']}")
    print("\\nGenerated Artifacts:")
    for artifact in summary['deployment_artifacts']:
        print(f"  ✅ {artifact}")
    
    print("\\n📋 Pre-deployment Checklist:")
    for item in summary['pre_deployment_checklist']:
        print(f"  ☐ {item}")
    
    print(f"\\n🎯 Ready for production deployment!")