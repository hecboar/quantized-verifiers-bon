"""Minimal RunPod REST client + SSH helpers (endpoints verified against
https://rest.runpod.io/v1/openapi.json on 2026-09-27).

Workflow per phase (scripts/pod.py): create pod -> wait for SSH -> upload code
tarball -> run bootstrap (self-stopping, hard timeout) -> download outputs ->
delete pod. Secrets are passed as pod env vars and never printed.
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass, field

BASE = "https://rest.runpod.io/v1"
DEFAULT_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"  # verify tag in phase 0


@dataclass
class PodSpec:
    name: str
    gpu_type: str = "NVIDIA GeForce RTX 4090"
    image: str = DEFAULT_IMAGE
    cloud_type: str = "SECURE"
    container_disk_gb: int = 40
    volume_gb: int = 80
    env: dict = field(default_factory=dict, repr=False)

    def body(self) -> dict:
        return {
            "name": self.name,
            "imageName": self.image,
            "gpuTypeIds": [self.gpu_type],
            "gpuCount": 1,
            "cloudType": self.cloud_type,
            "containerDiskInGb": self.container_disk_gb,
            "volumeInGb": self.volume_gb,
            "volumeMountPath": "/workspace",
            "ports": ["22/tcp"],
            "env": dict(self.env),
        }


class RunPodClient:
    def __init__(self, api_key: str, session=None, timeout: int = 60):
        if session is None:
            import requests

            session = requests.Session()
        self._key, self.s, self.timeout = api_key, session, timeout

    def __repr__(self) -> str:
        return "RunPodClient(<key hidden>)"

    def _req(self, method: str, path: str, body: dict | None = None) -> dict:
        r = self.s.request(method, BASE + path, headers={"Authorization": f"Bearer {self._key}"},
                           json=body, timeout=self.timeout)
        r.raise_for_status()
        try:
            return r.json()
        except ValueError:
            return {}

    def create_pod(self, spec: PodSpec) -> dict:
        return self._req("POST", "/pods", spec.body())

    def get_pod(self, pod_id: str) -> dict:
        return self._req("GET", f"/pods/{pod_id}")

    def list_pods(self) -> list:
        return self._req("GET", "/pods")

    def stop_pod(self, pod_id: str) -> dict:
        return self._req("POST", f"/pods/{pod_id}/stop")

    def delete_pod(self, pod_id: str) -> dict:
        return self._req("DELETE", f"/pods/{pod_id}")


def ssh_endpoint(pod: dict) -> tuple[str, int] | None:
    ip = pod.get("publicIp")
    port = (pod.get("portMappings") or {}).get("22")
    return (ip, int(port)) if ip and port else None


_SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "ServerAliveInterval=30"]


def ssh_argv(ep: tuple[str, int], command: str, key: str) -> list[str]:
    return ["ssh", *_SSH_OPTS, "-i", key, "-p", str(ep[1]), f"root@{ep[0]}", command]


def scp_argv(ep: tuple[str, int], src: str, dst: str, key: str, upload: bool) -> list[str]:
    remote = lambda p: f"root@{ep[0]}:{p}"
    a, b = (src, remote(dst)) if upload else (remote(src), dst)
    return ["scp", *_SSH_OPTS, "-i", key, "-P", str(ep[1]), "-r", a, b]


def bootstrap_command(cli_args: str, max_hours: float, workdir: str = "/workspace/vq") -> str:
    """Runs one CLI step on the pod with a hard timeout, logs to disk, and
    stops the pod afterwards whatever happens (no idle billing)."""
    return (
        f"cd {workdir} && "
        f"(pip install -q -e '.[gpu]' > /workspace/pip.log 2>&1; "
        f"timeout {int(max_hours)}h python -m vq.cli {cli_args} > /workspace/run.log 2>&1; "
        f"echo $? > /workspace/exit_code; "
        f"runpodctl stop pod $RUNPOD_POD_ID)"
    )


def remote_detached(cmd: str) -> str:
    return f"nohup bash -c {shlex.quote(cmd)} > /dev/null 2>&1 &"
