#!/bin/bash
# Rollback script for pg-neo-graph-rl

set -euo pipefail

ENVIRONMENT="${ENVIRONMENT:-production}"
PREVIOUS_VERSION="${1:-}"

if [ -z "$PREVIOUS_VERSION" ]; then
    echo "Usage: $0 <previous_version>"
    echo "Example: $0 v1.2.3"
    exit 1
fi

echo "🔄 Rolling back to version: $PREVIOUS_VERSION"

# Rollback Kubernetes deployment
kubectl rollout undo deployment/federated-graph-rl -n pg-neo-graph-rl

# Wait for rollback to complete
kubectl rollout status deployment/federated-graph-rl -n pg-neo-graph-rl --timeout=600s

# Verify rollback
kubectl wait --for=condition=ready pod -l app=federated-graph-rl -n pg-neo-graph-rl --timeout=300s

echo "✅ Rollback completed successfully"
