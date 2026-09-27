# Source me: exports the pod's environment into the current shell.
# RunPod injects pod env vars (the ones we pass at creation, plus RUNPOD_POD_ID and
# a pod-scoped RUNPOD_API_KEY) into the container's PID 1 only; SSH sessions do not
# inherit them. Without this, HF caches land on the 40 GB container disk and gated
# models are downloaded without HF_TOKEN.
while IFS= read -r -d '' kv; do
  case "$kv" in
    HF_HOME=*|HF_TOKEN=*|PIP_CACHE_DIR=*|RUNPOD_POD_ID=*|RUNPOD_API_KEY=*) export "$kv" ;;
  esac
done < /proc/1/environ
mkdir -p "${HF_HOME:-/workspace/hf_cache}" "${PIP_CACHE_DIR:-/workspace/pip_cache}"
