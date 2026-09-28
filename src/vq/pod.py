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
    # one GPU type for every phase (throughput and numerics stay comparable):
    # RTX 4090 COMMUNITY, 0.34 USD/h on 2026-09-27; stock fluctuates, so
    # creation retries while no instance is available
    gpu_type: str = "NVIDIA GeForce RTX 4090"
    image: str = DEFAULT_IMAGE
    cloud_type: str = "COMMUNITY"
    container_disk_gb: int = 40  # OS + pip-installed packages (vLLM, torch)
    # /workspace volume, deleted with the pod: every checkpoint of the study
    # (~55 GB: 3 generators + 3 Skywork ORMs + PRM-7B) + packed exports
    # (~8 GB) + outputs + pip cache. ~0.014 USD/h at 0.10 USD/GB/month.
    volume_gb: int = 100
    env: dict = field(default_factory=dict, repr=False)

    # caches on the volume, never on the 40 GB container disk
    CACHE_ENV = {"HF_HOME": "/workspace/hf_cache", "PIP_CACHE_DIR": "/workspace/pip_cache"}

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
            # host requirements (first real launches: a COMMUNITY host without a
            # public IP never exposed SSH; defaults would give 8 GB RAM per GPU)
            "supportPublicIp": True,
            "minRAMPerGPU": 24,
            "minVCPUPerGPU": 4,
            "minDownloadMbps": 200,
            # vllm==0.30.0 -> torch==2.13.0 (CUDA 13.0): the host driver must support 13.0
            "allowedCudaVersions": ["13.0"],
            "env": {**self.CACHE_ENV, **self.env},
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
        import time

        import requests

        # idempotent calls (GET / DELETE / stop) retry on transient network errors;
        # POST /pods is handled by create_pod_when_available (ambiguous outcome)
        attempts = 1 if (method == "POST" and path == "/pods") else 4
        for i in range(attempts):
            try:
                r = self.s.request(method, BASE + path, headers={"Authorization": f"Bearer {self._key}"},
                                   json=body, timeout=self.timeout)
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                if i == attempts - 1:
                    raise
                time.sleep(5 * (i + 1))
        if r.status_code >= 400:
            body = getattr(r, "text", "")[:500]
            raise RuntimeError(f"RunPod API {method} {path} -> HTTP {r.status_code}: {body}")
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


# RTX 4090 list prices on 2026-09-27 (USD/h) converted with margin for the volume and FX
_EUR_PER_HOUR = {"COMMUNITY": 0.34, "SECURE": 0.72}


def hourly_eur(cloud: str) -> float:
    return _EUR_PER_HOUR[cloud]


INSTALL_FETCH_HOURS = 5 / 60  # upload, fetch and delete after the grace period


def bootstrap_command(cli_args: str, max_hours: float, workdir: str = "/workspace/vq", grace_minutes: int = 20,
                      extras: str = "gpu") -> str:
    """One CLI step on the pod. A single hard cap (with forced kill) bounds
    install + run; a failed install stops immediately (exit 97). Afterwards the
    pod stays up `grace_minutes` so `watch` can fetch outputs over SSH
    (restarting a stopped pod can fail if its GPU was reassigned), then stops
    itself. The controller (`watch`) also enforces a deadline."""
    if "'" in cli_args:
        raise ValueError("step arguments must not contain single quotes")
    minutes = int(round(max_hours * 60))
    inner = (f". scripts/pod_env.sh; pip install -q -e \".[{extras}]\" > /workspace/pip.log 2>&1 "
             f"|| {{ echo 97 > /workspace/exit_code; exit 97; }}; "
             f"mkdir -p outputs/env; pip freeze > outputs/env/pip_freeze.txt 2>/dev/null; "
             f"nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > outputs/env/gpu.csv 2>&1; "
             f"python -m vq.cli {cli_args} > /workspace/run.log 2>&1; echo $? > /workspace/exit_code")
    return (
        f"cd {workdir} && "
        f"(timeout -k 120 {minutes}m bash -c '{inner}'; "
        f"[ -f /workspace/exit_code ] || echo 124 > /workspace/exit_code; "
        f"sleep {int(grace_minutes) * 60}; "
        f"bash scripts/self_stop.sh)"
    )


def billed_hours_cap(max_hours: float, grace_minutes: int = 20) -> float:
    """Worst-case billed time of one launch (cap + grace + transfer)."""
    return max_hours + grace_minutes / 60 + INSTALL_FETCH_HOURS


NO_STOCK = "no instances currently available"


def unique_pod_name(step: str) -> str:
    """Pod names identify a launch, so a pod created behind a dropped
    connection can be found (and adopted) instead of orphaned."""
    import uuid

    return f"vq-{step.split()[0]}-{uuid.uuid4().hex[:6]}"


def _find_pod_by_name(client: RunPodClient, name: str):
    try:
        pods = client.list_pods()
    except Exception:
        return None
    return next((p for p in pods or [] if p.get("name") == name), None)


def create_pod_when_available(client: RunPodClient, spec: PodSpec, retry_s: int = 120, max_wait_s: int = 6 * 3600,
                              sleep=None) -> dict:
    """Retry creation only while RunPod reports no free instance of this GPU
    (costs nothing while waiting); any other API error is raised at once."""
    import time

    import requests

    sleep = sleep or time.sleep
    waited = 0
    while True:
        try:
            return client.create_pod(spec)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            # ambiguous: the server may have created the pod before the connection died
            existing = _find_pod_by_name(client, spec.name)
            if existing is not None:
                print(f"connection dropped but pod {existing['id']} exists: adopting it", flush=True)
                return existing
            if waited >= max_wait_s:
                raise
            print(f"network error creating the pod ({type(e).__name__}); retrying in 30s", flush=True)
            sleep(30)
            waited += 30
        except RuntimeError as e:
            if NO_STOCK not in str(e) or waited >= max_wait_s:
                raise
            print(f"no {spec.gpu_type} available ({spec.cloud_type}); retrying in {retry_s}s "
                  f"(waited {waited // 60} of {max_wait_s // 60} min)", flush=True)
            sleep(retry_s)
            waited += max(retry_s, 1)


class BadHost(RuntimeError):
    """The pod's host is unfit (e.g. a power-capped GPU); delete it and try another."""


GPU_POWER_QUERY = "nvidia-smi --query-gpu=power.limit,power.default_limit --format=csv,noheader"


def gpu_power_ok(csv: str, min_frac: float = 0.8) -> tuple[bool, str]:
    """`power.limit, power.default_limit` -> fit? A COMMUNITY 4090 capped at 150 W
    of 450 W ran at 855 MHz and scored ~3x slower than the pilot host."""
    try:
        limit, default = (float(x.strip().split()[0]) for x in csv.strip().splitlines()[0].split(","))
    except (ValueError, IndexError):
        return False, f"unreadable GPU power query: {csv.strip()[:80]!r}"
    if limit < min_frac * default:
        return False, f"GPU power capped at {limit:.0f} W of {default:.0f} W"
    return True, f"GPU power limit {limit:.0f} W of {default:.0f} W"


def launch_pod(client: RunPodClient, spec: PodSpec, setup, wait_s: int = 900, poll_s: int = 10,
               retry_s: int = 120, max_stock_wait_s: int = 6 * 3600, max_bad_hosts: int = 5) -> str:
    """Create a pod and run `setup(pod_id, ssh_endpoint)` (host check, upload,
    start the step). Any failure before the step is running deletes the pod, so
    no GPU is left billing without a self-stopping job; a
    `BadHost` from setup deletes it and tries another host."""
    import time

    for attempt in range(1, max_bad_hosts + 1):
        pod = create_pod_when_available(client, spec, retry_s=retry_s, max_wait_s=max_stock_wait_s)
        pid = pod["id"]
        try:
            ep, waited = None, 0
            while True:
                ep = ssh_endpoint(client.get_pod(pid))
                if ep or waited >= wait_s:
                    break
                time.sleep(poll_s)
                waited += max(poll_s, 1)
            if not ep:
                raise RuntimeError("SSH never became available")
            setup(pid, ep)
        except BadHost as e:
            client.delete_pod(pid)
            print(f"pod {pid} rejected ({e}); deleted, trying another host ({attempt}/{max_bad_hosts})", flush=True)
            if attempt == max_bad_hosts:
                raise
            continue
        except BaseException:
            client.delete_pod(pid)
            raise
        return pid
    raise AssertionError("unreachable")


def parse_probe(raw: bytes | None) -> tuple[bool, list[str]]:
    """`cat exit_code; tail run.log` output -> (finished, lines). Decoded as
    UTF-8 with replacement: the log carries progress bars (Windows' default
    cp1252 decoding crashed the first overnight watch)."""
    if not raw:
        return False, []
    lines = raw.decode("utf-8", errors="replace").strip().splitlines()
    return (bool(lines) and lines[0].strip().isdigit()), lines


def bundle_code(repo, out, extras: list[str]) -> "Path":
    """Tarball for the pod: tracked files exactly as committed (git archive:
    normalised LF, no uncommitted edits; the pod runs the committed code) plus
    untracked data/output paths listed in `extras`. Refuses a dirty tree."""
    import subprocess
    import tarfile
    from pathlib import Path

    repo, out = Path(repo), Path(out)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo,
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError("uncommitted changes, commit before launching: " + dirty.replace("\n", "; "))
    import io

    out.parent.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(["git", "archive", "--format=tar", "HEAD"], cwd=repo, check=True,
                             capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as src, tarfile.open(out, "w:gz") as dst:
        for m in src.getmembers():
            if m.type in (tarfile.XGLTYPE, tarfile.XHDTYPE):  # pax headers: metadata only
                continue
            dst.addfile(m, src.extractfile(m) if m.isfile() else None)
        for extra in extras:
            p = repo / extra
            if p.exists():
                dst.add(p, arcname=extra)
    return out


def finish_pod(client: RunPodClient, pod_id: str, fetch):
    """Fetch outputs, then delete the pod no matter what happened while
    fetching (in the first real pilot a failed fetch left the pod billing)."""
    try:
        return fetch(client, pod_id)
    finally:
        client.delete_pod(pod_id)


def remote_archive_command(workdir: str) -> str:
    """Archive the pod's outputs; a step that failed before writing any output
    still yields a (possibly empty) archive instead of a tar error."""
    return f"cd {workdir} && mkdir -p outputs && tar czf /workspace/out.tar.gz outputs"


# steps allowed before the phase-1 gate has passed (cheap / validation only)
_PRE_GATE = ("pilot", "phase1", "gate", "annotate", "select-lambda", "export-packed", "parity", "plan",
             "prepare-data")


def requires_gate(step: str) -> bool:
    parts = step.split()
    cmd = parts[0]
    if cmd in _PRE_GATE:
        return False
    if cmd == "generate":
        return "--split" in parts and parts[parts.index("--split") + 1] not in ("dev", "gate")
    if cmd == "score":
        return "--dev-only" not in parts
    return True


def gate_allows(gate_file) -> bool:
    import json
    from pathlib import Path

    p = Path(gate_file)
    if not p.exists():
        return False
    try:
        return bool(json.loads(p.read_text()).get("go"))
    except ValueError:
        return False


def _campaign_key(rel: str) -> str | None:
    """store/<kind>/<name>.parquet|.json|.shards/... -> store/<kind>/<name>"""
    parts = rel.split("/")
    if len(parts) >= 3 and parts[0] == "store":
        name = parts[2]
        for suf in (".shards", ".parquet", ".json"):
            if name.endswith(suf):
                name = name[: -len(suf)]
                break
        return "/".join(parts[:2] + [name])
    return None


def merge_outputs(src_root, dst_root) -> dict:
    """Merge a pod's outputs into the local tree. Per-run reports are updated;
    other files are copied only when new or identical. A campaign (all files of
    store/<kind>/<name>) with ANY conflicting file is quarantined as a unit, so
    remote shards never land next to an incompatible local fingerprint."""
    import filecmp
    import fnmatch
    import shutil
    from pathlib import Path

    run_reports = ("pilot_*.json", "signal.json", "parity_*.json", "throughput.json", "env/*")
    src_root, dst_root = Path(src_root), Path(dst_root)
    rep = {"copied": [], "identical": [], "updated": [], "conflicts": [], "quarantined": []}
    files = sorted(p for p in src_root.rglob("*") if p.is_file())
    status = {}
    for f in files:
        rel = f.relative_to(src_root).as_posix()
        dst = dst_root / rel
        if not dst.exists():
            status[rel] = "new"
        elif filecmp.cmp(f, dst, shallow=False):
            status[rel] = "identical"
        elif any(fnmatch.fnmatch(rel, pat) for pat in run_reports):
            status[rel] = "update"
        else:
            status[rel] = "conflict"
    bad = {_campaign_key(r) for r, st in status.items() if st == "conflict" and _campaign_key(r)}
    rep["quarantined"] = sorted(bad)
    for f in files:
        rel = f.relative_to(src_root).as_posix()
        st, dst = status[rel], dst_root / rel
        if st == "conflict":
            rep["conflicts"].append(rel)
        elif _campaign_key(rel) in bad:
            continue  # part of a quarantined campaign: left in the pod snapshot only
        elif st == "new":
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            rep["copied"].append(rel)
        elif st == "identical":
            rep["identical"].append(rel)
        else:
            shutil.copy2(f, dst)
            rep["updated"].append(rel)
    return rep


def remote_detached(cmd: str) -> str:
    return f"nohup bash -c {shlex.quote(cmd)} > /dev/null 2>&1 &"
