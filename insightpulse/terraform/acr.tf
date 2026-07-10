# =============================================================================
# Azure Container Registry
# =============================================================================
# Holds the insightpulse-app and insightpulse-dashboard images built by CI.
# AKS pulls via its kubelet managed identity (AcrPull role) — no registry
# passwords anywhere in the cluster.
# =============================================================================

resource "azurerm_container_registry" "main" {
  # ACR names: alphanumeric only, globally unique.
  name                = replace("acr${local.name_prefix}", "-", "")
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = var.acr_sku

  # CI authenticates via OIDC service principal, never the admin account.
  admin_enabled = false

  # Premium-only hardening applied automatically when the SKU allows it.
  dynamic "retention_policy_in_days" {
    for_each = var.acr_sku == "Premium" ? [1] : []
    content {}
  }

  tags = local.common_tags
}

# Allow the AKS kubelets to pull images without imagePullSecrets.
resource "azurerm_role_assignment" "aks_acr_pull" {
  scope                            = azurerm_container_registry.main.id
  role_definition_name             = "AcrPull"
  principal_id                     = azurerm_kubernetes_cluster.main.kubelet_identity[0].object_id
  skip_service_principal_aad_check = true
}
