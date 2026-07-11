# InsightPulse — Azure Infrastructure (Terraform)

Provisions the complete production environment on Azure: an AKS cluster
(system + autoscaled user node pools), a container registry, PostgreSQL
Flexible Server (private access), Azure Cache for Redis, a Key Vault holding
every secret, and Log Analytics with Container Insights and alerting.

```
Internet ──► Ingress (AKS, snet-aks)
                │
     ┌──────────┴──────────┐
     │ insightpulse-app    │──► PostgreSQL  (snet-postgres, private DNS)
     │ insightpulse-dash   │──► Redis       (private endpoint, snet-endpoints)
     └──────────┬──────────┘
                │  Key Vault CSI driver (managed identity)
                ▼
            Key Vault  (LLM keys, DB password, Redis key)
```

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Terraform | ≥ 1.7 | infrastructure provisioning |
| Azure CLI | ≥ 2.60 | authentication, secret injection |
| kubectl | matching AKS | cluster operations after provisioning |

You also need an Azure subscription with `Owner` (or `Contributor` +
`User Access Administrator` — role assignments are created here).

## One-time setup: remote state

Terraform state contains resource ids (not secrets, thanks to Key Vault),
but still belongs in a locked, versioned backend:

```bash
az group create --name rg-insightpulse-tfstate --location centralindia
az storage account create --name stinsightpulsetf \
  --resource-group rg-insightpulse-tfstate --sku Standard_LRS \
  --allow-blob-public-access false
az storage container create --name tfstate --account-name stinsightpulsetf
```

## Deploy

```bash
cd terraform
az login && az account set --subscription "<subscription-id>"

# 1. Initialize with the remote backend (key is per-environment)
terraform init \
  -backend-config="resource_group_name=rg-insightpulse-tfstate" \
  -backend-config="storage_account_name=stinsightpulsetf" \
  -backend-config="container_name=tfstate" \
  -backend-config="key=insightpulse.dev.tfstate"

# 2. Review the plan for the chosen environment
terraform plan -var-file=environments/dev.tfvars -out=tfplan

# 3. Apply
terraform apply tfplan

# 4. Inject the LLM provider keys (never stored in Terraform state)
az keyvault secret set --vault-name "$(terraform output -raw key_vault_name)" \
  --name anthropic-api-key --value "$ANTHROPIC_API_KEY"
az keyvault secret set --vault-name "$(terraform output -raw key_vault_name)" \
  --name openai-api-key --value "$OPENAI_API_KEY"

# 5. Configure kubectl and continue with k8s/ (see docs/deployment.md)
$(terraform output -raw aks_get_credentials_command)
```

For production, repeat with `key=insightpulse.prod.tfstate` and
`-var-file=environments/prod.tfvars`.

## Environments

| Aspect | dev | prod |
|---|---|---|
| AKS control plane | Free | Standard (SLA) |
| User pool | 1-2 × D2s_v5 | 2-6 × D4s_v5 |
| PostgreSQL | B1ms, no HA, 7-day PITR | D2s_v3, zone-redundant HA, 35-day PITR |
| Redis | Basic C0 | Standard C1 (replicated) |
| Log retention | 30 days | 90 days |

## Security posture

- **No public data endpoints** — PostgreSQL is VNet-delegated, Redis sits
  behind a private endpoint, Key Vault denies by default.
- **No passwords in code or state outputs** — the DB password is generated
  by Terraform and lives only in Key Vault; workloads mount secrets via the
  AKS CSI driver with a managed identity; ACR pulls use the kubelet
  identity (admin account disabled).
- **Everything tagged** (`project`, `environment`, `owner`, `managed_by`)
  for cost attribution.

## Teardown

```bash
terraform destroy -var-file=environments/dev.tfvars
```

Note: with `purge_protection_enabled` (prod), the Key Vault survives in a
soft-deleted state for 30 days by design.
