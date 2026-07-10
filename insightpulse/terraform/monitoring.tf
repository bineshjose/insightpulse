# =============================================================================
# Monitoring: Log Analytics, Container Insights, and alert rules
# =============================================================================
# One workspace receives AKS control-plane logs, container stdout/stderr
# (the app's structlog JSON lines), and platform metrics. Alerts page the
# action group on the two failure modes that matter most: API pods
# crash-looping and node CPU saturation.
# =============================================================================

resource "azurerm_log_analytics_workspace" "main" {
  name                = "log-${local.name_prefix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = var.log_retention_days
  tags                = local.common_tags
}

# Control-plane diagnostics (API server, scheduler, autoscaler decisions).
resource "azurerm_monitor_diagnostic_setting" "aks" {
  name                       = "diag-${local.name_prefix}-aks"
  target_resource_id         = azurerm_kubernetes_cluster.main.id
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id

  enabled_log {
    category = "kube-apiserver"
  }

  enabled_log {
    category = "cluster-autoscaler"
  }

  enabled_metric {
    category = "AllMetrics"
  }
}

resource "azurerm_monitor_action_group" "oncall" {
  name                = "ag-${local.name_prefix}-oncall"
  resource_group_name = azurerm_resource_group.main.name
  short_name          = "ipulse"
  tags                = local.common_tags

  email_receiver {
    name          = "owner"
    email_address = var.alert_email
  }
}

# Node CPU saturation: sustained >85% means the HPA has hit max replicas
# or the user pool needs a larger max_count.
resource "azurerm_monitor_metric_alert" "node_cpu" {
  name                = "alert-${local.name_prefix}-node-cpu"
  resource_group_name = azurerm_resource_group.main.name
  scopes              = [azurerm_kubernetes_cluster.main.id]
  description         = "AKS node CPU above 85% for 15 minutes"
  severity            = 2
  frequency           = "PT5M"
  window_size         = "PT15M"
  tags                = local.common_tags

  criteria {
    metric_namespace = "Microsoft.ContainerService/managedClusters"
    metric_name      = "node_cpu_usage_percentage"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = 85
  }

  action {
    action_group_id = azurerm_monitor_action_group.oncall.id
  }
}

# Pod restarts: crash-looping API pods are the primary availability risk.
resource "azurerm_monitor_metric_alert" "pod_restarts" {
  name                = "alert-${local.name_prefix}-pod-restarts"
  resource_group_name = azurerm_resource_group.main.name
  scopes              = [azurerm_kubernetes_cluster.main.id]
  description         = "Container restarts detected in the last 15 minutes"
  severity            = 1
  frequency           = "PT5M"
  window_size         = "PT15M"
  tags                = local.common_tags

  criteria {
    metric_namespace = "Microsoft.ContainerService/managedClusters"
    metric_name      = "kube_pod_status_ready"
    aggregation      = "Average"
    operator         = "LessThan"
    threshold        = 1
  }

  action {
    action_group_id = azurerm_monitor_action_group.oncall.id
  }
}
