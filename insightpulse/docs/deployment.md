# Deployment Guide

Three tiers, in increasing fidelity: local development, Docker Compose
demo, and AKS production. Each tier uses the same code — only the `ENV`
profile (and therefore the layer strategies) changes.

## 1. Local development (no containers, no keys)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU wheel, no CUDA
pip install -e ".[dev]"

make generate-data          # 500 panelists, 10K purchases, 2.5K responses
make test                   # full suite (mocked externals)
make experiments            # the four thesis experiments

# Run the pieces directly:
uvicorn insightpulse.main:app --reload      # API on :8000
streamlit run dashboard/app.py              # dashboard on :8501
```

Demo mode (`ENV=demo`, the default) uses the CSV repository and the
simulated generation engine, so **every endpoint and dashboard page works
without any API key**.

## 2. Docker Compose demo

```bash
cp .env.example .env        # optionally add LLM keys for --live experiments
make generate-data          # sample data is volume-mounted into containers
make demo                   # builds both images, starts app + dashboard
```

| Service | URL | Health |
|---|---|---|
| API | http://localhost:8000 (docs at `/docs`) | `/health` |
| Dashboard | http://localhost:8501 | `/_stcore/health` |

Images are multi-stage, non-root, CPU-only-torch builds; see `Dockerfile`
and `Dockerfile.streamlit`.

## 3. AKS production

### 3.1 Provision infrastructure

Follow [terraform/README.md](../terraform/README.md): remote state,
`terraform apply -var-file=environments/prod.tfvars`, then inject the LLM
keys into Key Vault out-of-band.

### 3.2 One-time cluster add-ons

```bash
$(terraform -chdir=terraform output -raw aks_get_credentials_command)

# Ingress controller + certificate automation
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx --create-namespace
helm repo add jetstack https://charts.jetstack.io
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager --create-namespace --set crds.enabled=true
kubectl apply -f - <<'EOF'
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    privateKeySecretRef: {name: letsencrypt-prod-key}
    solvers: [{http01: {ingress: {ingressClassName: nginx}}}]
EOF
```

### 3.3 Render Key Vault identifiers into the CSI manifest

```bash
export KEYVAULT_NAME=$(terraform -chdir=terraform output -raw key_vault_name)
export KEYVAULT_CSI_CLIENT_ID=$(terraform -chdir=terraform output -raw keyvault_csi_client_id)
export TENANT_ID=$(terraform -chdir=terraform output -raw tenant_id)
envsubst < k8s/keyvault-csi.yaml | kubectl apply -f -
```

### 3.4 Deploy

Releases are cut by tagging main with a semver tag — CI/CD does the rest
(build → ACR → rollout → smoke test; see `ci/workflows/cd.yml`):

```bash
git tag v1.0.0 && git push origin v1.0.0
```

Manual first-time apply (equivalent to what CD runs):

```bash
kubectl apply -f k8s/namespace.yaml -f k8s/configmap.yaml
kubectl apply -f k8s/app-deployment.yaml -f k8s/app-service.yaml
kubectl apply -f k8s/dashboard-deployment.yaml -f k8s/dashboard-service.yaml
kubectl apply -f k8s/hpa.yaml -f k8s/ingress.yaml
kubectl -n insightpulse rollout status deployment/insightpulse-app
```

### 3.5 Verify

```bash
kubectl -n insightpulse get pods,hpa,ingress
kubectl -n insightpulse run smoke --rm -i --restart=Never \
  --image=curlimages/curl:8.9.1 -- curl -fsS http://insightpulse-app:8000/health
# structlog JSON lines flow into Log Analytics (Container Insights):
#   KubePodInventory | where Namespace == "insightpulse"
```

## Environment matrix

| Aspect | local dev | Docker demo | AKS production |
|---|---|---|---|
| `ENV` | demo | demo | production |
| L1 data | CSV | CSV | PostgreSQL (CSV fallback) |
| L3 generation | simulated | simulated | LLM providers |
| Secrets | none needed | `.env` (optional) | Key Vault via CSI |
| Scaling | n/a | single container | HPA 2-10 replicas |
| Logs | console | container stdout | Log Analytics |

## Rollback

Images are immutable and semver-tagged. Roll back by re-deploying the
previous tag:

```bash
kubectl -n insightpulse set image deployment/insightpulse-app \
  app=$ACR_LOGIN_SERVER/insightpulse-app:<previous-version>
kubectl -n insightpulse rollout status deployment/insightpulse-app
```
