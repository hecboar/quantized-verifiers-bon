#!/bin/bash
# Stop this pod through RunPod's GraphQL API with the pod-scoped key.
# Verified on 2026-09-27: runpodctl is unconfigured in the image and the REST API
# answers 403 to the pod key, while GraphQL accepts it.
cd "$(dirname "$0")/.." || exit 1
. scripts/pod_env.sh
for i in 1 2 3 4 5; do
  r=$(curl -s https://api.runpod.io/graphql \
        -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
        -d "{\"query\":\"mutation { podStop(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) { id desiredStatus } }\"}")
  echo "$(date -u) podStop attempt $i: $r" >> /workspace/self_stop.log
  echo "$r" | grep -q '"desiredStatus"' && exit 0
  sleep 60
done
exit 1
