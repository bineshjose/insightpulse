# =============================================================================
# Azure Key Vault — the single home for every secret
# =============================================================================
# Stores LLM provider keys, the generated Postgres password, and the Redis
# access key. Workloads read secrets through the AKS Key Vault CSI driver
# (see k8s/keyvault-csi.yaml) using the cluster's secrets-provider
# identity — secrets never appear in Terraform outputs, CI logs, or
# Kubernetes manifests.
# =============================================================================

data "azurerm_client_config" "current" {}

resource "azurerm_key_vault" "main" {
  name                = "kv-${local.name_prefix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"

  # RBAC instead of access policies: assignments are auditable and
  # revocable through standard Azure role tooling.
  rbac_authorization_enabled = true

  purge_protection_enabled   = var.environment == "prod"
  soft_delete_retention_days = 30

  network_acls {
    default_action = "Deny"
    bypass         = "AzureServices"
    virtual_network_subnet_ids = [
      azurerm_subnet.aks.id,
      azurerm_subnet.endpoints.id,
    ]
    # Terraform apply needs the operator's egress IP allowed.
    ip_rules = var.keyvault_allowed_ips
  }

  tags = local.common_tags
}

# The identity running `terraform apply` manages secret contents.
resource "azurerm_role_assignment" "deployer_kv_admin" {
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Administrator"
  principal_id         = data.azurerm_client_config.current.object_id
}

# The AKS CSI-driver identity reads secrets at pod mount time.
resource "azurerm_role_assignment" "aks_kv_secrets_user" {
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_kubernetes_cluster.main.key_vault_secrets_provider[0].secret_identity[0].object_id
}

# -----------------------------------------------------------------------------
# Managed secrets (values generated or referenced, never hardcoded)
# -----------------------------------------------------------------------------

resource "azurerm_key_vault_secret" "postgres_password" {
  name         = "postgres-admin-password"
  value        = random_password.postgres_admin.result
  key_vault_id = azurerm_key_vault.main.id

  depends_on = [azurerm_role_assignment.deployer_kv_admin]
}

resource "azurerm_key_vault_secret" "database_url" {
  name = "database-url"
  value = format(
    "postgresql+asyncpg://%s:%s@%s:5432/%s",
    var.postgres_admin_login,
    random_password.postgres_admin.result,
    azurerm_postgresql_flexible_server.main.fqdn,
    var.postgres_database_name,
  )
  key_vault_id = azurerm_key_vault.main.id

  depends_on = [azurerm_role_assignment.deployer_kv_admin]
}

resource "azurerm_key_vault_secret" "redis_url" {
  name = "redis-url"
  value = format(
    "rediss://:%s@%s:6380/0",
    azurerm_redis_cache.main.primary_access_key,
    azurerm_redis_cache.main.hostname,
  )
  key_vault_id = azurerm_key_vault.main.id

  depends_on = [azurerm_role_assignment.deployer_kv_admin]
}

# LLM provider keys are pushed out-of-band (never through Terraform state):
#   az keyvault secret set --vault-name kv-insightpulse-prod \
#     --name anthropic-api-key --value "$ANTHROPIC_API_KEY"
# The placeholders below only reserve the names the CSI driver expects.
resource "azurerm_key_vault_secret" "anthropic_api_key" {
  name         = "anthropic-api-key"
  value        = "set-me-out-of-band"
  key_vault_id = azurerm_key_vault.main.id

  lifecycle {
    # Real value is set via `az keyvault secret set`; don't revert it.
    ignore_changes = [value]
  }

  depends_on = [azurerm_role_assignment.deployer_kv_admin]
}

resource "azurerm_key_vault_secret" "openai_api_key" {
  name         = "openai-api-key"
  value        = "set-me-out-of-band"
  key_vault_id = azurerm_key_vault.main.id

  lifecycle {
    ignore_changes = [value]
  }

  depends_on = [azurerm_role_assignment.deployer_kv_admin]
}
