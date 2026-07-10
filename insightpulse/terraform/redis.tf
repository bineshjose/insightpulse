# =============================================================================
# Azure Cache for Redis
# =============================================================================
# Production cache backend (CacheBackend.REDIS): embedding lookups, survey
# result caching, and rate-limit counters. TLS-only; the access key is
# surfaced to workloads exclusively through Key Vault.
# =============================================================================

resource "azurerm_redis_cache" "main" {
  name                = "redis-${local.name_prefix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  capacity = var.redis_capacity
  family   = var.redis_family
  sku_name = var.redis_sku_name

  non_ssl_port_enabled = false
  minimum_tls_version  = "1.2"

  redis_configuration {
    # LRU eviction: cached distributions are recomputable, never precious.
    maxmemory_policy = "allkeys-lru"
  }

  tags = local.common_tags
}

# Private endpoint keeps cache traffic off the public internet.
resource "azurerm_private_endpoint" "redis" {
  name                = "pe-${local.name_prefix}-redis"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  subnet_id           = azurerm_subnet.endpoints.id
  tags                = local.common_tags

  private_service_connection {
    name                           = "psc-redis"
    private_connection_resource_id = azurerm_redis_cache.main.id
    subresource_names              = ["redisCache"]
    is_manual_connection           = false
  }
}
