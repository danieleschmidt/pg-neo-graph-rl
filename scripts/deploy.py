#!/usr/bin/env python3
"""Deployment automation script for PG-Neo-Graph-RL."""

import argparse
import subprocess
import sys
import os
import json
import yaml
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import shutil


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_step(message: str):
    """Print a deployment step message."""
    print(f"{Colors.OKBLUE}[DEPLOY]{Colors.ENDC} {message}")


def print_success(message: str):
    """Print a success message."""
    print(f"{Colors.OKGREEN}[SUCCESS]{Colors.ENDC} {message}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"{Colors.WARNING}[WARNING]{Colors.ENDC} {message}")


def print_error(message: str):
    """Print an error message."""
    print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {message}")


def run_command(cmd: List[str], cwd: str = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print_step(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            check=check, 
            capture_output=True, 
            text=True
        )
        if result.stdout and result.stdout.strip():
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {' '.join(cmd)}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        if check:
            sys.exit(1)
        return e


def check_prerequisites():
    """Check if required tools are available."""
    print_step("Checking prerequisites...")
    
    required_tools = {
        "docker": "Docker for containerization",
        "kubectl": "Kubernetes CLI (optional)",
        "helm": "Helm package manager (optional)"
    }
    
    missing_tools = []
    for tool, description in required_tools.items():
        try:
            run_command([tool, "--version"], check=False)
        except FileNotFoundError:
            if tool in ["kubectl", "helm"]:
                print_warning(f"{tool} not found - {description}")
            else:
                missing_tools.append(tool)
    
    if missing_tools:
        print_error(f"Missing required tools: {', '.join(missing_tools)}")
        sys.exit(1)
    
    print_success("Prerequisites check completed")


def create_kubernetes_manifests(namespace: str = "pg-neo-graph-rl"):
    """Create Kubernetes deployment manifests."""
    print_step("Creating Kubernetes manifests...")
    
    k8s_dir = Path("k8s")
    k8s_dir.mkdir(exist_ok=True)
    
    # Namespace
    namespace_manifest = {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {
            "name": namespace,
            "labels": {
                "app": "pg-neo-graph-rl"
            }
        }
    }
    
    # ConfigMap
    configmap_manifest = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": "pg-neo-graph-rl-config",
            "namespace": namespace
        },
        "data": {
            "LOG_LEVEL": "INFO",
            "ENVIRONMENT": "production",
            "JAX_PLATFORM_NAME": "cpu",
            "NUM_AGENTS": "10",
            "COMMUNICATION_PROTOCOL": "gossip"
        }
    }
    
    # Deployment
    deployment_manifest = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "pg-neo-graph-rl",
            "namespace": namespace,
            "labels": {
                "app": "pg-neo-graph-rl"
            }
        },
        "spec": {
            "replicas": 3,
            "selector": {
                "matchLabels": {
                    "app": "pg-neo-graph-rl"
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": "pg-neo-graph-rl"
                    }
                },
                "spec": {
                    "containers": [
                        {
                            "name": "pg-neo-graph-rl",
                            "image": "pg-neo-graph-rl:production-latest",
                            "ports": [
                                {
                                    "containerPort": 8000,
                                    "name": "http"
                                }
                            ],
                            "envFrom": [
                                {
                                    "configMapRef": {
                                        "name": "pg-neo-graph-rl-config"
                                    }
                                }
                            ],
                            "resources": {
                                "requests": {
                                    "memory": "1Gi",
                                    "cpu": "500m"
                                },
                                "limits": {
                                    "memory": "2Gi",
                                    "cpu": "1000m"
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }
                    ]
                }
            }
        }
    }
    
    # Service
    service_manifest = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": "pg-neo-graph-rl-service",
            "namespace": namespace,
            "labels": {
                "app": "pg-neo-graph-rl"
            }
        },
        "spec": {
            "selector": {
                "app": "pg-neo-graph-rl"
            },
            "ports": [
                {
                    "protocol": "TCP",
                    "port": 80,
                    "targetPort": 8000,
                    "name": "http"
                }
            ],
            "type": "ClusterIP"
        }
    }
    
    # Write manifests
    manifests = {
        "namespace.yaml": namespace_manifest,
        "configmap.yaml": configmap_manifest,
        "deployment.yaml": deployment_manifest,
        "service.yaml": service_manifest
    }
    
    for filename, manifest in manifests.items():
        with open(k8s_dir / filename, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
    
    print_success("Kubernetes manifests created")


def create_helm_chart():
    """Create Helm chart for deployment."""
    print_step("Creating Helm chart...")
    
    chart_dir = Path("helm/pg-neo-graph-rl")
    chart_dir.mkdir(parents=True, exist_ok=True)
    templates_dir = chart_dir / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    # Chart.yaml
    chart_yaml = {
        "apiVersion": "v2",
        "name": "pg-neo-graph-rl",
        "description": "Federated Graph-Neural Reinforcement Learning toolkit",
        "type": "application",
        "version": "0.1.0",
        "appVersion": "1.0.0",
        "maintainers": [
            {
                "name": "Daniel Schmidt",
                "email": "daniel@example.com"
            }
        ],
        "keywords": [
            "machine-learning",
            "reinforcement-learning",
            "federated-learning",
            "graph-neural-networks"
        ]
    }
    
    # values.yaml
    values_yaml = {
        "replicaCount": 3,
        "image": {
            "repository": "pg-neo-graph-rl",
            "tag": "production-latest",
            "pullPolicy": "IfNotPresent"
        },
        "service": {
            "type": "ClusterIP",
            "port": 80,
            "targetPort": 8000
        },
        "resources": {
            "requests": {
                "memory": "1Gi",
                "cpu": "500m"
            },
            "limits": {
                "memory": "2Gi",
                "cpu": "1000m"
            }
        },
        "env": {
            "LOG_LEVEL": "INFO",
            "ENVIRONMENT": "production",
            "JAX_PLATFORM_NAME": "cpu",
            "NUM_AGENTS": "10",
            "COMMUNICATION_PROTOCOL": "gossip"
        },
        "autoscaling": {
            "enabled": False,
            "minReplicas": 2,
            "maxReplicas": 10,
            "targetCPUUtilizationPercentage": 80
        },
        "monitoring": {
            "enabled": True,
            "serviceMonitor": {
                "enabled": True,
                "port": "metrics"
            }
        }
    }
    
    # Write chart files
    with open(chart_dir / "Chart.yaml", "w") as f:
        yaml.dump(chart_yaml, f, default_flow_style=False, sort_keys=False)
    
    with open(chart_dir / "values.yaml", "w") as f:
        yaml.dump(values_yaml, f, default_flow_style=False, sort_keys=False)
    
    print_success("Helm chart created")


def create_docker_compose_production():
    """Create production Docker Compose configuration."""
    print_step("Creating production Docker Compose...")
    
    compose_config = {
        "version": "3.8",
        "services": {
            "pg-neo-graph-rl": {
                "build": {
                    "context": ".",
                    "target": "production"
                },
                "environment": [
                    "LOG_LEVEL=INFO",
                    "ENVIRONMENT=production",
                    "JAX_PLATFORM_NAME=cpu"
                ],
                "ports": [
                    "8000:8000"
                ],
                "restart": "unless-stopped",
                "healthcheck": {
                    "test": ["CMD", "python", "-c", "import pg_neo_graph_rl; print('OK')"],
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 3,
                    "start_period": "40s"
                }
            },
            "prometheus": {
                "image": "prom/prometheus:latest",
                "ports": [
                    "9090:9090"
                ],
                "volumes": [
                    "./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml"
                ],
                "restart": "unless-stopped"
            },
            "grafana": {
                "image": "grafana/grafana:latest",
                "ports": [
                    "3000:3000"
                ],
                "environment": [
                    "GF_SECURITY_ADMIN_PASSWORD=admin"
                ],
                "volumes": [
                    "grafana-data:/var/lib/grafana"
                ],
                "restart": "unless-stopped"
            }
        },
        "volumes": {
            "grafana-data": None
        },
        "networks": {
            "pg-neo-network": {
                "driver": "bridge"
            }
        }
    }
    
    with open("docker-compose.prod.yml", "w") as f:
        yaml.dump(compose_config, f, default_flow_style=False, sort_keys=False)
    
    print_success("Production Docker Compose created")


def deploy_to_kubernetes(namespace: str = "pg-neo-graph-rl", dry_run: bool = False):
    """Deploy to Kubernetes cluster."""
    print_step("Deploying to Kubernetes...")
    
    k8s_dir = Path("k8s")
    if not k8s_dir.exists():
        print_error("Kubernetes manifests not found. Run with --create-manifests first.")
        sys.exit(1)
    
    cmd_args = ["kubectl", "apply", "-f", str(k8s_dir)]
    if dry_run:
        cmd_args.append("--dry-run=client")
    
    run_command(cmd_args)
    
    if not dry_run:
        # Wait for deployment to be ready
        print_step("Waiting for deployment to be ready...")
        run_command([
            "kubectl", "rollout", "status", 
            f"deployment/pg-neo-graph-rl", 
            "-n", namespace,
            "--timeout=300s"
        ])
    
    print_success("Kubernetes deployment completed")


def deploy_with_helm(namespace: str = "pg-neo-graph-rl", dry_run: bool = False):
    """Deploy using Helm."""
    print_step("Deploying with Helm...")
    
    chart_dir = Path("helm/pg-neo-graph-rl")
    if not chart_dir.exists():
        print_error("Helm chart not found. Run with --create-helm first.")
        sys.exit(1)
    
    cmd = [
        "helm", "upgrade", "--install", "pg-neo-graph-rl",
        str(chart_dir),
        "--namespace", namespace,
        "--create-namespace"
    ]
    
    if dry_run:
        cmd.append("--dry-run")
    
    run_command(cmd)
    
    print_success("Helm deployment completed")


def deploy_docker_compose():
    """Deploy using Docker Compose."""
    print_step("Deploying with Docker Compose...")
    
    if not Path("docker-compose.prod.yml").exists():
        print_error("Production Docker Compose file not found.")
        sys.exit(1)
    
    run_command([
        "docker-compose", "-f", "docker-compose.prod.yml", 
        "up", "-d", "--build"
    ])
    
    print_success("Docker Compose deployment completed")


def health_check(url: str = "http://localhost:8000/health"):
    """Perform health check on deployed service."""
    print_step(f"Performing health check on {url}...")
    
    import urllib.request
    import urllib.error
    
    max_retries = 30
    retry_delay = 10
    
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    print_success("Health check passed")
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, Exception) as e:
            print_warning(f"Health check attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print_step(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    
    print_error("Health check failed after all retries")
    return False


def main():
    """Main deployment script."""
    parser = argparse.ArgumentParser(description="Deployment automation for PG-Neo-Graph-RL")
    
    parser.add_argument(
        "platform",
        choices=["kubernetes", "helm", "docker-compose", "all"],
        help="Deployment platform"
    )
    
    parser.add_argument(
        "--namespace",
        default="pg-neo-graph-rl",
        help="Kubernetes namespace"
    )
    
    parser.add_argument(
        "--create-manifests",
        action="store_true",
        help="Create Kubernetes manifests"
    )
    
    parser.add_argument(
        "--create-helm",
        action="store_true",
        help="Create Helm chart"
    )
    
    parser.add_argument(
        "--create-compose",
        action="store_true",
        help="Create production Docker Compose"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run without making changes"
    )
    
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip health check after deployment"
    )
    
    args = parser.parse_args()
    
    try:
        check_prerequisites()
        
        # Create deployment artifacts if requested
        if args.create_manifests or args.platform == "kubernetes":
            create_kubernetes_manifests(args.namespace)
        
        if args.create_helm or args.platform == "helm":
            create_helm_chart()
        
        if args.create_compose or args.platform == "docker-compose":
            create_docker_compose_production()
        
        # Deploy based on platform
        if args.platform == "kubernetes":
            deploy_to_kubernetes(args.namespace, args.dry_run)
        elif args.platform == "helm":
            deploy_with_helm(args.namespace, args.dry_run)
        elif args.platform == "docker-compose":
            deploy_docker_compose()
        elif args.platform == "all":
            # Create all artifacts
            create_kubernetes_manifests(args.namespace)
            create_helm_chart()
            create_docker_compose_production()
            print_success("All deployment artifacts created")
        
        # Health check
        if not args.skip_health_check and not args.dry_run and args.platform != "all":
            health_check()
        
        print_success("Deployment completed successfully!")
        
    except KeyboardInterrupt:
        print_error("Deployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()