# =============================================================================
# InsightPulse — Azure Infrastructure (Terraform)
# =============================================================================
# Root module: provider configuration, resource group, and the AKS cluster.
#
# Layout (one concern per file, HashiCorp style):
#   main.tf        — providers, resource group, AKS
#   network.tf     — VNet, subnets, NSGs
#   acr.tf         — container registry + AKS pull permission
#   postgres.tf    — PostgreSQL Flexible Server (private access)
#   redis.tf       — Azure Cache for Redis
#   keyvault.tf    — Key Vault + CSI-driver access for AKS workloads
#   monitoring.tf  — Log Analytics + Container Insights + alerts
#   variables.tf   — every knob, documented, with sane defaults
#   outputs.tf     — endpoints and credentials consumed by CI/CD
#
# State: keep remote. The backend block is parameterized at `terraform init`
# time (backend config must be static, so values live in the init command):
#
#   terraform init \
#     -backend-config="resource_group_name=rg-insightpulse-tfstate" \
#     -backend-config="storage_account_name=stinsightpulsetf" \
#     -backend-config="container_name=tfstate" \
#     -backend-config="key=insightpulse.<env>.tfstate"
# =============================================================================

terraform {
  required_version = ">= 1.7.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  backend "azurerm" {}
}

provider "azurerm" {
  features {
    key_vault {
      # Soft-delete protection: never purge a vault from Terraform.
      purge_soft_delete_on_destroy    = false
      recover_soft_deleted_key_vaults = true
    }
  }
}

# -----------------------------------------------------------------------------
# Naming and tagging conventions (applied to every resource)
# -----------------------------------------------------------------------------

locals {
  # e.g. "insightpulse-prod"
  name_prefix = "${var.project_name}-${var.environment}"

  common_tags = {
    project     = var.project_name
    environment = var.environment
    managed_by  = "terraform"
    owner       = var.owner
  }
}

# -----------------------------------------------------------------------------
# Resource group
# -----------------------------------------------------------------------------

resource "azurerm_resource_group" "main" {
  name     = "rg-${local.name_prefix}"
  location = var.location
  tags     = local.common_tags
}

# -----------------------------------------------------------------------------
# AKS cluster
#   - system pool: cluster-critical pods only (CriticalAddonsOnly taint)
#   - user pool:   application workloads, autoscaled
#   - managed identity + Azure CNI + Log Analytics integration
# -----------------------------------------------------------------------------

resource "azurerm_kubernetes_cluster" "main" {
  name                = "aks-${local.name_prefix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = local.name_prefix
  kubernetes_version  = var.kubernetes_version
  sku_tier            = var.aks_sku_tier

  default_node_pool {
    name                 = "system"
    vm_size              = var.system_node_vm_size
    node_count           = var.system_node_count
    vnet_subnet_id       = azurerm_subnet.aks.id
    only_critical_addons_enabled = true

    upgrade_settings {
      max_surge = "33%"
    }
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin    = "azure"
    network_policy    = "azure"
    load_balancer_sku = "standard"
    service_cidr      = var.aks_service_cidr
    dns_service_ip    = var.aks_dns_service_ip
  }

  oms_agent {
    log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  }

  key_vault_secrets_provider {
    # CSI driver rotates mounted secrets without pod restarts.
    secret_rotation_enabled  = true
    secret_rotation_interval = var.keyvault_secret_rotation_interval
  }

  azure_policy_enabled = true

  tags = local.common_tags
}

# Application workloads run on the autoscaled user pool.
resource "azurerm_kubernetes_cluster_node_pool" "user" {
  name                  = "user"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size               = var.user_node_vm_size
  vnet_subnet_id        = azurerm_subnet.aks.id
  mode                  = "User"

  auto_scaling_enabled = true
  min_count            = var.user_node_min_count
  max_count            = var.user_node_max_count

  node_labels = {
    "workload" = "insightpulse"
  }

  tags = local.common_tags
}
