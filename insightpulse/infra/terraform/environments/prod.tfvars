# =============================================================================
# Production environment — SLA-backed, zone-redundant, longer retention
# =============================================================================
# Usage:
#   terraform plan -var-file=environments/prod.tfvars

environment = "prod"
location    = "centralindia"

# --- AKS: SLA-backed control plane, autoscaled general-purpose pool ---
aks_sku_tier        = "Standard"
system_node_vm_size = "Standard_D2s_v5"
system_node_count   = 2
user_node_vm_size   = "Standard_D4s_v5"
user_node_min_count = 2
user_node_max_count = 6

# --- Data services: replicated tiers, HA database ---
acr_sku                        = "Standard"
postgres_sku_name              = "GP_Standard_D2s_v3"
postgres_storage_mb            = 131072
postgres_ha_enabled            = true
postgres_backup_retention_days = 35
redis_sku_name                 = "Standard"
redis_family                   = "C"
redis_capacity                 = 1

# --- Observability ---
log_retention_days = 90
