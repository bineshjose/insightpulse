# =============================================================================
# Outputs — consumed by CI/CD (cd.yml) and operators
# =============================================================================
# Secrets are never output: the Postgres password, Redis key, and provider
# API keys live in Key Vault only. CI reads the identifiers below to build,
# push, and deploy.
# =============================================================================

output "resource_group_name" {
  description = "Resource group holding every project resource."
  value       = azurerm_resource_group.main.name
}

output "aks_cluster_name" {
  description = "AKS cluster name (for `az aks get-credentials`)."
  value       = azurerm_kubernetes_cluster.main.name
}

output "aks_get_credentials_command" {
  description = "Copy-paste command to configure kubectl."
  value = format(
    "az aks get-credentials --resource-group %s --name %s",
    azurerm_resource_group.main.name,
    azurerm_kubernetes_cluster.main.name,
  )
}

output "acr_login_server" {
  description = "Registry host CI pushes images to (e.g. acrinsightpulseprod.azurecr.io)."
  value       = azurerm_container_registry.main.login_server
}

output "key_vault_name" {
  description = "Key Vault holding all application secrets."
  value       = azurerm_key_vault.main.name
}

output "key_vault_uri" {
  description = "Key Vault URI referenced by the CSI SecretProviderClass."
  value       = azurerm_key_vault.main.vault_uri
}

output "keyvault_csi_client_id" {
  description = "Client id of the AKS secrets-provider identity (k8s/keyvault-csi.yaml)."
  value       = azurerm_kubernetes_cluster.main.key_vault_secrets_provider[0].secret_identity[0].client_id
}

output "postgres_fqdn" {
  description = "Private FQDN of the PostgreSQL server."
  value       = azurerm_postgresql_flexible_server.main.fqdn
}

output "redis_hostname" {
  description = "Redis host (TLS port 6380; key lives in Key Vault)."
  value       = azurerm_redis_cache.main.hostname
}

output "log_analytics_workspace_id" {
  description = "Workspace receiving cluster and application logs."
  value       = azurerm_log_analytics_workspace.main.id
}

output "tenant_id" {
  description = "Azure AD tenant id (needed by the CSI SecretProviderClass)."
  value       = data.azurerm_client_config.current.tenant_id
}
