# Kubernetes manifests

```
k8s/
├── infra/local-path-storage.yaml   # StorageClass (Rancher's K3s ships without one)
├── secrets.example.env             # template of the Secret mlops-secrets
└── base/                           # namespace mlops: Postgres, MinIO, MLflow (Kustomize)
```

## Ports

The cluster is K3s running *inside* the Rancher container, so a NodePort is
only reachable through ports that container publishes. The VM only allows
8000–8025, so the container maps host 8010–8020 to NodePorts 30010–30020:

```bash
docker run -d --name rancher --privileged --restart=unless-stopped \
  -p 8002:80 -p 8001:443 -p 8010-8020:30010-30020 \
  -v rancher_data:/var/lib/rancher rancher/rancher:v2.15.2
```

| VM port | NodePort | Service |
|---|---|---|
| 8001 | — | Rancher (HTTPS) |
| 8002 | — | Rancher (HTTP → redirects to HTTPS) |
| 8010 | 30010 | Airflow UI |
| 8011 | 30011 | MLflow UI |
| 8012 | 30012 | MinIO console |
| 8013 | 30013 | Inference API |
| 8014 | 30014 | Streamlit |
| 8015 | 30015 | Grafana |
| 8016 | 30016 | Prometheus |
| 8017 | 30017 | Locust UI |
| 8018–8020 | 30018–30020 | spare |

Postgres and the MinIO S3 API are internal only.

## Apply

```bash
kubectl create namespace mlops
kubectl -n mlops create secret generic mlops-secrets --from-env-file=k8s/secrets.env
kubectl apply -k k8s/base
```
