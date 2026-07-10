# =============================================================================
# Input variables — every configurable knob, documented, with sane defaults
# =============================================================================
# Environment-specific values live in environments/<env>.tfvars; anything
# not overridden there falls back to the defaults below (sized for dev).
# =============================================================================

# -----------------------------------------------------------------------------
# Project identity
# -----------------------------------------------------------------------------

variable "project_name" {
  description = "Project slug used in every resource name."
  type        = string
  default     = "insightpulse"

  validation {
    condition     = can(regex("^[a-z][a-z0-9]{2,15}$", var.project_name))
    error_message = "project_name must be 3-16 lowercase alphanumeric characters (ACR/KV name limits)."
  }
}

variable "environment" {
  description = "Deployment environment (drives sizing, HA, and retention)."
  type        = string

  validation {
    condition     = contains(["dev", "prod"], var.environment)
    error_message = "environment must be 'dev' or 'prod'."
  }
}

variable "location" {
  description = "Azure region for all resources."
  type        = string
  default     = "centralindia"
}

variable "owner" {
  description = "Owner tag applied to every resource (person or team)."
  type        = string
  default     = "binesh-jose"
}

variable "alert_email" {
  description = "Email address for monitor alerts."
  type        = string
  default     = "ch24m521@smail.iitm.ac.in"
}

# -----------------------------------------------------------------------------
# AKS
# -----------------------------------------------------------------------------

variable "kubernetes_version" {
  description = "AKS Kubernetes version (null = latest stable in the region)."
  type        = string
  default     = null
}

variable "aks_sku_tier" {
  description = "AKS control-plane tier: Free (dev) or Standard (SLA-backed)."
  type        = string
  default     = "Free"

  validation {
    condition     = contains(["Free", "Standard"], var.aks_sku_tier)
    error_message = "aks_sku_tier must be 'Free' or 'Standard'."
  }
}

variable "system_node_vm_size" {
  description = "VM size for the system node pool (cluster-critical pods)."
  type        = string
  default     = "Standard_D2s_v5"
}

variable "system_node_count" {
  description = "Fixed node count for the system pool."
  type        = number
  default     = 1
}

variable "user_node_vm_size" {
  description = "VM size for the user (application) node pool."
  type        = string
  default     = "Standard_D4s_v5"
}

variable "user_node_min_count" {
  description = "Autoscaler floor for the user pool."
  type        = number
  default     = 1
}

variable "user_node_max_count" {
  description = "Autoscaler ceiling for the user pool."
  type        = number
  default     = 3
}

variable "aks_service_cidr" {
  description = "CIDR for Kubernetes ClusterIP services (must not overlap the VNet)."
  type        = string
  default     = "10.20.0.0/16"
}

variable "aks_dns_service_ip" {
  description = "Cluster DNS IP (must sit inside aks_service_cidr)."
  type        = string
  default     = "10.20.0.10"
}

# -----------------------------------------------------------------------------
# Networking
# -----------------------------------------------------------------------------

variable "vnet_address_space" {
  description = "Address space for the project VNet."
  type        = string
  default     = "10.10.0.0/16"
}

variable "aks_subnet_prefix" {
  description = "Subnet for AKS nodes and pods (Azure CNI)."
  type        = string
  default     = "10.10.0.0/20"
}

variable "postgres_subnet_prefix" {
  description = "Delegated subnet for PostgreSQL Flexible Server."
  type        = string
  default     = "10.10.16.0/24"
}

variable "endpoints_subnet_prefix" {
  description = "Subnet for private endpoints (Redis, Key Vault, ACR)."
  type        = string
  default     = "10.10.17.0/24"
}

# -----------------------------------------------------------------------------
# Container registry
# -----------------------------------------------------------------------------

variable "acr_sku" {
  description = "ACR tier; Premium adds retention policies and private link."
  type        = string
  default     = "Basic"

  validation {
    condition     = contains(["Basic", "Standard", "Premium"], var.acr_sku)
    error_message = "acr_sku must be Basic, Standard, or Premium."
  }
}

# -----------------------------------------------------------------------------
# PostgreSQL
# -----------------------------------------------------------------------------

variable "postgres_version" {
  description = "PostgreSQL major version."
  type        = string
  default     = "16"
}

variable "postgres_sku_name" {
  description = "Flexible Server compute SKU."
  type        = string
  default     = "B_Standard_B1ms"
}

variable "postgres_storage_mb" {
  description = "Provisioned storage in MB."
  type        = number
  default     = 32768
}

variable "postgres_admin_login" {
  description = "Administrator login name (password is generated + vaulted)."
  type        = string
  default     = "insightpulse_admin"
}

variable "postgres_database_name" {
  description = "Application database name."
  type        = string
  default     = "insightpulse"
}

variable "postgres_backup_retention_days" {
  description = "Point-in-time-restore window."
  type        = number
  default     = 7
}

variable "postgres_ha_enabled" {
  description = "Zone-redundant high availability (prod only; doubles cost)."
  type        = bool
  default     = false
}

# -----------------------------------------------------------------------------
# Redis
# -----------------------------------------------------------------------------

variable "redis_sku_name" {
  description = "Redis tier: Basic (dev), Standard (replicated), Premium (VNet)."
  type        = string
  default     = "Basic"
}

variable "redis_family" {
  description = "Redis family: C (Basic/Standard) or P (Premium)."
  type        = string
  default     = "C"
}

variable "redis_capacity" {
  description = "Redis size within the family (C0=250MB ... C6=53GB)."
  type        = number
  default     = 0
}

# -----------------------------------------------------------------------------
# Key Vault / observability
# -----------------------------------------------------------------------------

variable "keyvault_allowed_ips" {
  description = "Operator egress IPs allowed through the Key Vault firewall (for terraform apply)."
  type        = list(string)
  default     = []
}

variable "keyvault_secret_rotation_interval" {
  description = "How often the AKS CSI driver re-reads mounted secrets."
  type        = string
  default     = "2m"
}

variable "log_retention_days" {
  description = "Log Analytics retention window."
  type        = number
  default     = 30
}
