# Terraform variables for pg-neo-graph-rl
variable "environment" {
  description = "Deployment environment"
  type = string
  default = "production"
}

variable "region" {
  description = "AWS region"
  type = string
  default = "us-west-2"
}

variable "min_instances" {
  description = "Minimum number of instances"
  type = number
  default = 3
}

variable "max_instances" {
  description = "Maximum number of instances"
  type = number
  default = 50
}

variable "database_instance_class" {
  description = "RDS instance class"
  type = string
  default = "db.r5.large"
}

variable "enable_encryption" {
  description = "Enable encryption at rest"
  type = bool
  default = true
}
