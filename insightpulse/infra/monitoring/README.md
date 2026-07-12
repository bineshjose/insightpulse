# Monitoring Stack — Prometheus + Grafana on AKS

This directory contains the cluster-side monitoring assets for InsightPulse:

| File | Purpose |
| --- | --- |
| `prometheus/rules.yaml` | Alerting rules mirroring `insightpulse.observability.alerting` |
| `grafana/insightpulse-dashboard.json` | The platform operations dashboard |

The application exposes Prometheus metrics on `GET /metrics` (text
exposition format, rendered by `MetricsCollector.render_prometheus()`).
Install the `observability` extra in the production image so the
`prometheus_client` backend is used:

```bash
pip install "insightpulse[observability]"
```

## 1. Deploy kube-prometheus-stack

Use the community Helm chart, which bundles the Prometheus Operator,
Prometheus, Alertmanager, and Grafana:

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --set grafana.adminPassword="$(az keyvault secret show \
        --vault-name insightpulse-kv --name grafana-admin --query value -o tsv)" \
  --set prometheus.prometheusSpec.retention=15d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=20Gi
```

Notes for AKS:

- Use the `managed-premium` storage class for the Prometheus PVC on
  production node pools (`--set ...storageSpec...storageClassName=managed-premium`).
- If the cluster uses Azure Monitor managed Prometheus instead, the same
  `rules.yaml` expressions can be loaded as an
  `AzureManagedPrometheusRuleGroup`; the metric names are identical.

## 2. Enable scraping (ServiceMonitor)

The scrape config lives with the app manifests so it ships with every
release:

```bash
kubectl apply -f ../k8s/servicemonitor.yaml
```

The `release: kube-prometheus-stack` label on the ServiceMonitor must match
the Helm release name from step 1 — the operator's
`serviceMonitorSelector` only picks up matching objects. Verify discovery:

```bash
kubectl -n monitoring port-forward svc/kube-prometheus-stack-prometheus 9090
# open http://localhost:9090/targets and look for insightpulse/insightpulse-app
```

## 3. Load the alerting rules

Wrap `prometheus/rules.yaml` in a `PrometheusRule` custom resource so the
operator ships it to Prometheus:

```bash
kubectl create -n monitoring -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: insightpulse-alerts
  labels:
    release: kube-prometheus-stack
spec:
$(sed 's/^/  /' prometheus/rules.yaml)
EOF
```

Route notifications in Alertmanager by the `severity` label:
`critical` → PagerDuty, `warning` → Slack `#insightpulse-ops`,
`info` → Slack digest. The application-side `AlertManager`
(`insightpulse.observability.alerting`) fires the same rules in-process,
so keep thresholds in sync with `settings.observability` when editing.

## 4. Import the Grafana dashboard

Via the UI: **Dashboards → New → Import**, upload
`grafana/insightpulse-dashboard.json`, and select the Prometheus
datasource when prompted (the JSON references it as `${DS_PROMETHEUS}`).

Or provision it declaratively with a sidecar-watched ConfigMap:

```bash
kubectl create configmap insightpulse-dashboard \
  --namespace monitoring \
  --from-file=grafana/insightpulse-dashboard.json
kubectl label configmap insightpulse-dashboard \
  --namespace monitoring grafana_dashboard="1"
```

kube-prometheus-stack's Grafana sidecar auto-loads any ConfigMap carrying
the `grafana_dashboard` label.

## 5. Smoke test

```bash
# 1. Metrics endpoint answers with exposition text
kubectl -n insightpulse port-forward svc/insightpulse-app 8000 &
curl -s localhost:8000/metrics | head

# 2. Prometheus sees the target and the rules
kubectl -n monitoring port-forward svc/kube-prometheus-stack-prometheus 9090 &
curl -s 'localhost:9090/api/v1/rules' | jq '.data.groups[].name'

# 3. Grafana renders the dashboard
kubectl -n monitoring port-forward svc/kube-prometheus-stack-grafana 3000:80
# open http://localhost:3000/d/insightpulse-ops
```

Panel-by-panel interpretation of the dashboard is documented in
`docs/observability.md`.
