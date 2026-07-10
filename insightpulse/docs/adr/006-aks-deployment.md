# ADR-006: AKS for production deployment

## Status

Accepted (Stage 5).

## Context

Production needs: two long-running services (FastAPI pipeline, Streamlit
dashboard), burst-heavy compute (a survey run fans out hundreds of
concurrent LLM calls and CPU-bound calibration), managed PostgreSQL and
Redis, secret management, autoscaling, and an industry-credible
operations story for NielsenIQ evaluators. The organization context is
Azure (NIQ's cloud standard).

Options considered:

1. **Azure App Service** — simplest PaaS; but weak fit for multi-container
   topologies, no fine-grained autoscaling signals, and the 30-minute
   request ceiling is uncomfortably close to large survey runs.
2. **Azure Container Apps** — serverless containers with KEDA scaling;
   genuinely attractive for the API, but less control over node sizing
   (the transformer/FAISS workload is memory-hungry), immature Key Vault
   CSI story at design time, and no straightforward path to GPU pools if
   LoRA fine-tuning moves in-cluster.
3. **AKS** — full Kubernetes: node-pool control, HPA on custom metrics,
   Key Vault CSI driver, Azure CNI networking, and the deployment
   patterns (probes, security contexts, rolling updates) that an
   enterprise review expects to see demonstrated.

## Decision

Deploy to **AKS** with a system/user node-pool split, HPA on CPU plus a
requests-per-second custom metric, the Key Vault CSI driver for secrets,
and Terraform-managed infrastructure (see `terraform/`). CI/CD is
trunk-based: semver tags build images to ACR and roll out via `kubectl`
with rollout gates and a smoke test.

## Consequences

- (+) Node pools sized independently: cluster-critical pods isolated on
  the system pool, the app on an autoscaled user pool.
- (+) The manifests demonstrate production Kubernetes practice (restricted
  Pod Security, non-root, read-only root FS, resource limits) — part of
  the engineering story being evaluated.
- (+) A GPU node pool is an additive change if twin generation ever moves
  from API models to in-cluster LoRA-tuned models (thesis L3 roadmap).
- (−) Kubernetes carries real operational overhead versus Container Apps;
  accepted deliberately — the overhead *is* the demonstration — and
  mitigated by keeping the demo path on plain Docker Compose.
- (−) An idle AKS cluster costs money; dev tfvars minimize this (1-2
  small nodes, Free control plane) and `terraform destroy` is routine.
