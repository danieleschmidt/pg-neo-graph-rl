# Production Infrastructure for pg-neo-graph-rl
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }
  
  backend "s3" {
    bucket = "pg-neo-graph-rl-terraform-state"
    key    = "production/terraform.tfstate"
    region = "us-west-2"
    encrypt = true
    dynamodb_table = "terraform-state-lock"
  }
}

provider "aws" {
  region = "us-west-2"
  
  default_tags {
    tags = {
      Environment = "production"
      Project     = "pg-neo-graph-rl"
      ManagedBy   = "terraform"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

# VPC and Networking
module "vpc" {
  source = "./modules/vpc"
  
  environment = "production"
  cidr_block = "10.0.0.0/16"
  availability_zones = ["us-west-2a", "us-west-2b", "us-west-2c"]
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
  enable_dns_hostnames = true
  enable_dns_support = true
  
  tags = {
    Name = "pg-neo-graph-rl-vpc"
  }
}

# EKS Cluster
module "eks" {
  source = "./modules/eks"
  
  cluster_name = "pg-neo-graph-rl-production"
  cluster_version = "1.27"
  
  vpc_id = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  node_groups = {
    general = {
      instance_types = ["m5.large", "m5.xlarge"]
      capacity_type = "ON_DEMAND"
      min_size = 3
      max_size = 50
      desired_size = 4
    }
    compute = {
      instance_types = ["c5.2xlarge", "c5.4xlarge"]
      capacity_type = "SPOT"
      min_size = 0
      max_size = 50
      desired_size = 5
    }
    gpu = {
      instance_types = ["p3.2xlarge", "g4dn.xlarge"]
      capacity_type = "ON_DEMAND"
      min_size = 0
      max_size = 10
      desired_size = 0
    }
  }
  
  enable_irsa = true
}

# RDS Database
module "database" {
  source = "./modules/rds"
  
  identifier = "pg-neo-graph-rl-production"
  engine = "postgres"
  engine_version = "14.9"
  instance_class = "db.r5.large"
  
  allocated_storage = 100
  max_allocated_storage = 1000
  storage_encrypted = true
  
  vpc_id = module.vpc.vpc_id
  subnet_ids = module.vpc.database_subnets
  
  backup_retention_period = 30
  backup_window = "03:00-04:00"
  maintenance_window = "sun:04:00-sun:05:00"
}

# ElastiCache Redis
module "cache" {
  source = "./modules/elasticache"
  
  cluster_id = "pg-neo-graph-rl-production"
  node_type = "cache.r6g.large"
  num_cache_nodes = 3
  
  vpc_id = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
}

# Application Load Balancer
module "alb" {
  source = "./modules/alb"
  count = true ? 1 : 0
  
  name = "pg-neo-graph-rl-production"
  vpc_id = module.vpc.vpc_id
  subnet_ids = module.vpc.public_subnets
  
  certificate_arn = "arn:aws:acm:us-west-2:123456789012:certificate/12345678-1234-1234-1234-123456789012"
  domain_name = "federated-rl.example.com"
}

# Monitoring and Observability
module "monitoring" {
  source = "./modules/monitoring"
  
  cluster_name = module.eks.cluster_name
  vpc_id = module.vpc.vpc_id
  
  enable_prometheus = true
  enable_grafana = true
  enable_cloudwatch = true
  
  log_retention_days = 30
}

# Security
module "security" {
  source = "./modules/security"
  
  vpc_id = module.vpc.vpc_id
  cluster_name = module.eks.cluster_name
  
  enable_waf = true
  enable_secrets_manager = true
}

# Outputs
output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value = module.eks.cluster_endpoint
}

output "database_endpoint" {
  description = "RDS database endpoint"
  value = module.database.endpoint
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value = module.cache.endpoint
}

output "load_balancer_dns" {
  description = "Application Load Balancer DNS"
  value = true ? module.alb[0].dns_name : null
}
