# =============================================================================
# Development environment — minimal cost, no HA, short retention
# =============================================================================
# Usage:
#   terraform plan -var-file=environments/dev.tfvars

environment = "dev"
location    = "centralindia"

# --- AKS: single small node, free control plane ---
aks_sku_tier        = "Free"
system_node_vm_size = "Standard_D2s_v5"
system_node_count   = 1
user_node_vm_size   = "Standard_D2s_v5"
user_node_min_count = 1
user_node_max_count = 2

# --- Data services: burstable / basic tiers ---
acr_sku             = "Basic"
postgres_sku_name   = "B_Standard_B1ms"
postgres_storage_mb = 32768
postgres_ha_enabled = false
redis_sku_name      = "Basic"
redis_family        = "C"
redis_capacity      = 0

# --- Observability ---
log_retention_days = 30
