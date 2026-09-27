"""Run one pipeline step on a RunPod RTX 4090 and bring the results back.

  python scripts/pod.py launch --step "pilot --generator qwen2.5-1.5b" --hours 3          # dry run
  python scripts/pod.py launch --step "pilot --generator qwen2.5-1.5b" --hours 3 --yes    # real, costs money
  python scripts/pod.py watch  --pod <id>     # waits for the step, downloads outputs, deletes the pod
  python scripts/pod.py fetch  --pod <id>     # download outputs/ and logs without deleting
  python scripts/pod.py delete --pod <id>
  python scripts/pod.py list

Safety: dry run by default; the pod stops itself when the step ends (or at
the --hours hard timeout); secrets travel only as pod env vars.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vq.pod import PodSpec, RunPodClient, bootstrap_command, remote_detached, scp_argv, ssh_argv, ssh_endpoint  # noqa: E402

KEY = Path.home() / ".ssh" / "id_ed25519"
REMOTE = "/workspace/vq"


def _env():
    from dotenv import dotenv_values

    vals = dotenv_values(ROOT / ".env")
    for k in ("HF_TOKEN", "RUNPOD_API_KEY"):
        if not vals.get(k):
            sys.exit(f"{k} missing in .env")
    return vals


def _bundle() -> Path:
    """Code (tracked files) + data manifests + current outputs (for resumable phases)."""
    out = ROOT / "outputs" / "_bundle.tar.gz"
    out.parent.mkdir(exist_ok=True)
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    with tarfile.open(out, "w:gz") as tar:
        for f in tracked:
            if f != ".env":
                tar.add(ROOT / f, arcname=f)
        for extra in ("data", "outputs/store", "outputs/frozen_lambda.json", "outputs/throughput.json", "outputs/packed"):
            p = ROOT / extra
            if p.exists():
                tar.add(p, arcname=extra)
    return out


def _run(argv):
    print("$", " ".join(a if "@" not in a else a for a in argv[:6]), "...", flush=True)
    return subprocess.run(argv, check=True)


def cmd_launch(a):
    env = _env()
    pub = KEY.with_suffix(".pub")
    if not pub.exists():
        sys.exit(f"SSH key {pub} not found. Create one with: ssh-keygen -t ed25519 -f {KEY}")
    spec = PodSpec(name=f"vq-{a.step.split()[0]}", cloud_type=a.cloud,
                   env={"HF_TOKEN": env["HF_TOKEN"], "PUBLIC_KEY": pub.read_text().strip()})
    cmd = bootstrap_command(a.step, max_hours=a.hours, workdir=REMOTE)
    print(f"pod: {spec.gpu_type} ({spec.cloud_type}), image {spec.image}")
    print(f"step: python -m vq.cli {a.step}   (hard timeout {a.hours} h, self-stop at the end)")
    print(f"max cost at 0.69 EUR/h: {a.hours * 0.69:.2f} EUR")
    if not a.yes:
        print("DRY RUN: add --yes to create the pod.")
        return
    client = RunPodClient(env["RUNPOD_API_KEY"])
    pod = client.create_pod(spec)
    pid = pod["id"]
    print(f"created pod {pid}; waiting for SSH ...", flush=True)
    ep = None
    for _ in range(90):
        ep = ssh_endpoint(client.get_pod(pid))
        if ep:
            break
        time.sleep(10)
    if not ep:
        client.delete_pod(pid)
        sys.exit("SSH never became available; pod deleted.")
    time.sleep(20)  # sshd start-up
    bundle = _bundle()
    _run(ssh_argv(ep, f"mkdir -p {REMOTE}", str(KEY)))
    _run(scp_argv(ep, str(bundle), f"{REMOTE}/bundle.tar.gz", str(KEY), upload=True))
    _run(ssh_argv(ep, f"cd {REMOTE} && tar xzf bundle.tar.gz && rm bundle.tar.gz", str(KEY)))
    _run(ssh_argv(ep, remote_detached(cmd), str(KEY)))
    print(f"running. Next: python scripts/pod.py watch --pod {pid}")


def _fetch(client, pid):
    ep = ssh_endpoint(client.get_pod(pid))
    if not ep:
        sys.exit("pod has no SSH endpoint (stopped pods must be started to fetch; RunPod keeps the volume)")
    dst = ROOT / "outputs" / "pod_runs" / pid
    dst.mkdir(parents=True, exist_ok=True)
    for remote in ("/workspace/run.log", "/workspace/pip.log", "/workspace/exit_code"):
        subprocess.run(scp_argv(ep, remote, str(dst), str(KEY), upload=False))
    _run(ssh_argv(ep, f"cd {REMOTE} && tar czf /workspace/out.tar.gz outputs", str(KEY)))
    _run(scp_argv(ep, "/workspace/out.tar.gz", str(dst), str(KEY), upload=False))
    with tarfile.open(dst / "out.tar.gz") as tar:
        tar.extractall(ROOT, filter="data")
    print(f"outputs merged into {ROOT / 'outputs'}; logs in {dst}")


def cmd_watch(a):
    """Poll over SSH until the step writes /workspace/exit_code (the pod then
    idles for its grace period), fetch outputs, delete the pod."""
    env = _env()
    client = RunPodClient(env["RUNPOD_API_KEY"])
    while True:
        pod = client.get_pod(a.pod)
        status = pod.get("desiredStatus")
        ep = ssh_endpoint(pod)
        finished = False
        if status == "RUNNING" and ep:
            r = subprocess.run(ssh_argv(ep, "cat /workspace/exit_code 2>/dev/null; tail -n 2 /workspace/run.log 2>/dev/null",
                                        str(KEY)), capture_output=True, text=True)
            lines = r.stdout.strip().splitlines()
            finished = bool(lines) and lines[0].strip().isdigit()
            print(time.strftime("%H:%M:%S"), status, "|", " / ".join(lines[-2:])[:160], flush=True)
        else:
            print(time.strftime("%H:%M:%S"), status, flush=True)
            if status in ("EXITED", "STOPPED", "TERMINATED"):
                sys.exit("pod stopped before outputs were fetched: start it from the RunPod console, then run `fetch`.")
        if finished:
            break
        time.sleep(a.every)
    _fetch(client, a.pod)
    code = (ROOT / "outputs" / "pod_runs" / a.pod / "exit_code").read_text().strip()
    client.delete_pod(a.pod)
    print(f"step exit code {code}; pod {a.pod} deleted.")


def cmd_fetch(a):
    _fetch(RunPodClient(_env()["RUNPOD_API_KEY"]), a.pod)


def cmd_delete(a):
    RunPodClient(_env()["RUNPOD_API_KEY"]).delete_pod(a.pod)
    print("deleted")


def cmd_list(a):
    for p in RunPodClient(_env()["RUNPOD_API_KEY"]).list_pods():
        print(p.get("id"), p.get("name"), p.get("desiredStatus"), p.get("costPerHr"))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("launch")
    s.add_argument("--step", required=True, help='vq.cli arguments, e.g. "pilot --generator qwen2.5-1.5b"')
    s.add_argument("--hours", type=float, required=True)
    s.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    s.add_argument("--yes", action="store_true")
    s.set_defaults(fn=cmd_launch)
    for name, fn in (("watch", cmd_watch), ("fetch", cmd_fetch), ("delete", cmd_delete)):
        s = sub.add_parser(name)
        s.add_argument("--pod", required=True)
        if name == "watch":
            s.add_argument("--every", type=int, default=300)
        s.set_defaults(fn=fn)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
