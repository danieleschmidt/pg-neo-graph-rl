# 🚀 GitHub Actions Setup Guide

## Overview
This guide explains how to set up the CI/CD pipeline for pg-neo-graph-rl using GitHub Actions.

## Workflow File Location
Due to GitHub App permissions, the workflow file needs to be manually copied to the correct location:

**Source**: `docs/workflows/production-deployment.yml`  
**Destination**: `.github/workflows/production-deployment.yml`

## Setup Steps

### 1. Copy Workflow File
```bash
# Create .github/workflows directory if it doesn't exist
mkdir -p .github/workflows

# Copy the workflow file
cp docs/workflows/production-deployment.yml .github/workflows/production-deployment.yml
```

### 2. Configure Repository Secrets
Add the following secrets in your GitHub repository settings:

#### Required Secrets
- `AWS_ACCESS_KEY_ID`: AWS access key for deployment
- `AWS_SECRET_ACCESS_KEY`: AWS secret key for deployment

#### Optional Secrets (for enhanced features)
- `SLACK_WEBHOOK_URL`: For deployment notifications
- `DATADOG_API_KEY`: For metrics integration

### 3. Environment Configuration
Create a `production` environment in your repository settings with:
- Required reviewers (optional)
- Environment secrets (if different from repository secrets)
- Deployment protection rules

### 4. AWS IAM Permissions
Ensure your AWS credentials have the following permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "eks:*",
                "ec2:*",
                "rds:*",
                "elasticache:*",
                "iam:*",
                "s3:*",
                "cloudwatch:*",
                "logs:*"
            ],
            "Resource": "*"
        }
    ]
}
```

### 5. Terraform Backend Setup
Initialize the Terraform S3 backend:

```bash
# Create S3 bucket for Terraform state
aws s3 mb s3://pg-neo-graph-rl-terraform-state

# Create DynamoDB table for state locking
aws dynamodb create-table \
    --table-name terraform-state-lock \
    --attribute-definitions AttributeName=LockID,AttributeType=S \
    --key-schema AttributeName=LockID,KeyType=HASH \
    --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5
```

## Workflow Features

### 🧪 Automated Testing
- **Unit Tests**: PyTest with coverage reporting
- **Security Scanning**: Bandit for security vulnerabilities
- **Code Quality**: Ruff linting and Black formatting
- **Type Checking**: MyPy static analysis

### 🐳 Container Build
- **Multi-platform**: AMD64 and ARM64 support
- **Layer Caching**: GitHub Actions cache for faster builds
- **Security Scanning**: Container vulnerability assessment
- **Registry Push**: GitHub Container Registry (GHCR)

### 🚀 Deployment Pipeline
- **Infrastructure**: Terraform deployment to AWS
- **Application**: Kubernetes rolling deployment
- **Health Checks**: Automated smoke tests
- **Rollback**: Automatic rollback on failure

### 📊 Monitoring Integration
- **Metrics**: Automatic metrics collection
- **Alerts**: Integration with monitoring systems
- **Notifications**: Slack/Teams deployment updates

## Workflow Triggers

### Automatic Triggers
- **Push to main**: Deploys to production
- **Version tags**: `v*` tags trigger releases
- **Pull requests**: Runs tests only

### Manual Triggers
```bash
# Trigger workflow manually
gh workflow run production-deployment.yml
```

## Monitoring Deployments

### View Workflow Status
```bash
# List workflow runs
gh run list --workflow=production-deployment.yml

# View specific run
gh run view <run-id>

# Watch logs in real-time
gh run watch
```

### Kubernetes Monitoring
```bash
# Check deployment status
kubectl get deployments -n pg-neo-graph-rl

# View pod logs
kubectl logs -f deployment/federated-graph-rl -n pg-neo-graph-rl

# Check service health
kubectl get svc -n pg-neo-graph-rl
```

## Troubleshooting

### Common Issues

#### 1. AWS Permissions Error
```
Error: AccessDenied
```
**Solution**: Verify AWS IAM permissions and credentials

#### 2. Terraform State Lock
```
Error: Error locking state
```
**Solution**: Check DynamoDB table and release manual locks

#### 3. Kubernetes Connection
```
Error: Unable to connect to cluster
```
**Solution**: Verify EKS cluster exists and kubeconfig is updated

#### 4. Container Registry Permissions
```
Error: denied: permission_denied
```
**Solution**: Check GITHUB_TOKEN permissions for package registry

### Debug Commands

```bash
# Check workflow logs
gh run view --log

# Validate Terraform configuration
cd deployment/terraform
terraform validate

# Test Kubernetes connectivity
kubectl cluster-info

# Verify container image
docker pull ghcr.io/yourusername/pg-neo-graph-rl:latest
```

## Security Considerations

### Secret Management
- Use GitHub repository secrets for sensitive data
- Rotate credentials regularly
- Use least-privilege IAM policies
- Enable secret scanning in repository settings

### Container Security
- Regular base image updates
- Vulnerability scanning enabled
- Non-root user execution
- Read-only file systems where possible

### Network Security
- VPC with private subnets
- Security groups with minimal access
- TLS encryption for all communications
- Network policies in Kubernetes

## Performance Optimization

### Build Optimization
- Multi-stage Docker builds
- Layer caching strategies
- Parallel job execution
- Conditional workflow steps

### Deployment Speed
- Rolling updates with minimal downtime
- Health checks for faster validation
- Pre-pulled container images
- Resource pre-allocation

## Maintenance

### Regular Tasks
- Update workflow dependencies monthly
- Review and rotate secrets quarterly
- Monitor resource usage
- Update base images for security patches

### Monitoring
- Set up alerts for failed deployments
- Monitor deployment duration trends
- Track success/failure rates
- Review security scan results

---

**📝 Note**: This workflow provides a production-ready CI/CD pipeline with comprehensive testing, security scanning, and automated deployment capabilities.