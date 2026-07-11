# =============================================================================
# Azure Database for PostgreSQL — Flexible Server
# =============================================================================
# Production persistence for panel data (L1 SQLRepository). Private access
# only (VNet-integrated via the delegated subnet); the admin password is
# generated here and stored exclusively in Key Vault.
# =============================================================================

resource "random_password" "postgres_admin" {
  length  = 32
  special = false # avoids URL-encoding pitfalls in connection strings
}

resource "azurerm_postgresql_flexible_server" "main" {
  name                = "psql-${local.name_prefix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  version    = var.postgres_version
  sku_name   = var.postgres_sku_name
  storage_mb = var.postgres_storage_mb

  administrator_login    = var.postgres_admin_login
  administrator_password = random_password.postgres_admin.result

  # Private access: reachable from the VNet only, no public endpoint.
  delegated_subnet_id           = azurerm_subnet.postgres.id
  private_dns_zone_id           = azurerm_private_dns_zone.postgres.id
  public_network_access_enabled = false

  backup_retention_days        = var.postgres_backup_retention_days
  geo_redundant_backup_enabled = var.environment == "prod"

  dynamic "high_availability" {
    for_each = var.postgres_ha_enabled ? [1] : []
    content {
      mode = "ZoneRedundant"
    }
  }

  tags = local.common_tags

  depends_on = [azurerm_private_dns_zone_virtual_network_link.postgres]
}

resource "azurerm_postgresql_flexible_server_database" "insightpulse" {
  name      = var.postgres_database_name
  server_id = azurerm_postgresql_flexible_server.main.id
  charset   = "UTF8"
  collation = "en_US.utf8"
}
