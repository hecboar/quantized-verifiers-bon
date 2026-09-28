"""Run one pipeline step on a RunPod RTX 4090 and bring the results back.

  python scripts/pod.py launch --step "pilot --generator qwen2.5-1.5b" --hours 3          # dry run
  python scripts/pod.py launch --step "pilot --generator qwen2.5-1.5b" --hours 3 --yes    # real, costs money
  python scripts/pod.py watch  --pod <id>     # waits for the step, downloads outputs, deletes the pod
  python scripts/pod.py fetch  --pod <id>     # download + merge outputs/ and logs without deleting
  python scripts/pod.py delete --pod <id>
  python scripts/pod.py list

Safety:
- dry run by default; the printed cost is the worst case (hard cap + grace);
- one hard cap bounds install + run; a failed install stops at once;
- any failure between pod creation and the running step deletes the pod;
- grid steps are refused unless outputs/gate.json says go (or --force);
- `watch` enforces the deadline from this side too;
- fetched outputs are merged without ever overwriting a differing local file;
- secrets travel only as pod env vars.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vq.pod import (PodSpec, RunPodClient, billed_hours_cap, bootstrap_command, gate_allows,  # noqa: E402
                    finish_pod, launch_pod, merge_outputs, remote_archive_command, remote_detached, requires_gate,
                    scp_argv, ssh_argv, ssh_endpoint, unique_pod_name, hourly_eur, parse_probe, bundle_code)

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
    """Committed code (git archive, LF) + data manifests + current outputs."""
    extras = ["data", "outputs/store", "outputs/frozen_lambda.json", "outputs/throughput.json", "outputs/packed"]
    return bundle_code(ROOT, ROOT / "outputs" / "_bundle.tar.gz", extras)


def _run(argv):
    print("$", " ".join(argv[:6]), "...", flush=True)
    return subprocess.run(argv, check=True)


def _pod_record(pid: str) -> Path:
    return ROOT / "outputs" / "pods" / f"{pid}.json"


def cmd_launch(a):
    if a.yes:
        _keep_awake()  # from pod creation on: no sleep before the self-stop is armed
    env = _env()
    pub = KEY.with_suffix(".pub")
    if not pub.exists():
        sys.exit(f"SSH key {pub} not found. Create one with: ssh-keygen -t ed25519 -f {KEY}")
    if requires_gate(a.step) and not gate_allows(ROOT / "outputs" / "gate.json") and not a.force:
        sys.exit("refusing: this step spends grid budget and outputs/gate.json does not say go "
                 "(run the phase-1 gate first, or pass --force deliberately)")
    spec = PodSpec(name=unique_pod_name(a.step), cloud_type=a.cloud,
                   env={"HF_TOKEN": env["HF_TOKEN"], "PUBLIC_KEY": pub.read_text().strip()})
    cmd = bootstrap_command(a.step, max_hours=a.hours, workdir=REMOTE, extras=a.extras, grace_minutes=a.grace)
    cap = billed_hours_cap(a.hours, a.grace)
    print(f"pod: {spec.gpu_type} ({spec.cloud_type}), image {spec.image}")
    print(f"step: python -m vq.cli {a.step}")
    print(f"hard cap {a.hours} h (install + run) + grace {a.grace} min -> worst case {cap:.2f} h "
          f"= {cap * hourly_eur(a.cloud):.2f} EUR at {hourly_eur(a.cloud)} EUR/h ({a.cloud})")
    if not a.yes:
        print("DRY RUN: add --yes to create the pod.")
        return
    client = RunPodClient(env["RUNPOD_API_KEY"])

    def setup(pid, ep):
        # persist the pod id first: if this process dies, `list` shows it as ours
        rec = _pod_record(pid)
        rec.parent.mkdir(parents=True, exist_ok=True)
        rec.write_text(json.dumps({"pod": pid, "step": a.step, "created": time.time(),
                                   "deadline": time.time() + cap * 3600}))
        time.sleep(20)  # sshd start-up
        bundle = _bundle()
        _run(ssh_argv(ep, f"mkdir -p {REMOTE}", str(KEY)))
        _run(scp_argv(ep, str(bundle), f"{REMOTE}/bundle.tar.gz", str(KEY), upload=True))
        _run(ssh_argv(ep, f"cd {REMOTE} && tar xzf bundle.tar.gz && rm bundle.tar.gz", str(KEY)))
        _run(ssh_argv(ep, remote_detached(cmd), str(KEY)))

    # up to 25 min for SSH: a fresh host may need to pull the ~10 GB image first
    pid = launch_pod(client, spec, setup, wait_s=1500)  # deletes the pod if anything in setup fails
    price = client.get_pod(pid).get("costPerHr")
    print(f"running pod {pid} at {price} USD/h. Next: python scripts/pod.py watch --pod {pid}")
    if a.watch:  # no unattended gap between launch and watch
        cmd_watch(argparse.Namespace(pod=pid, every=180))


def _fetch(client, pid):
    ep = ssh_endpoint(client.get_pod(pid))
    if not ep:
        sys.exit("pod has no SSH endpoint (stopped pods must be started to fetch; RunPod keeps the volume)")
    dst = ROOT / "outputs" / "pod_runs" / pid
    dst.mkdir(parents=True, exist_ok=True)
    for remote in ("/workspace/run.log", "/workspace/pip.log", "/workspace/exit_code"):
        subprocess.run(scp_argv(ep, remote, str(dst), str(KEY), upload=False))
    _run(ssh_argv(ep, remote_archive_command(REMOTE), str(KEY)))
    _run(scp_argv(ep, "/workspace/out.tar.gz", str(dst), str(KEY), upload=False))
    snap = dst / "snapshot"
    with tarfile.open(dst / "out.tar.gz") as tar:
        tar.extractall(snap, filter="data")  # pod-specific directory, never straight into outputs/
    rep = merge_outputs(snap / "outputs", ROOT / "outputs")
    (dst / "merge_report.json").write_text(json.dumps(rep, indent=1))
    print(f"merged: {len(rep['copied'])} new, {len(rep['updated'])} updated reports, {len(rep['identical'])} identical, {len(rep['conflicts'])} conflicts")
    for c in rep["conflicts"]:
        print(f"  CONFLICT kept local version: {c} (pod version in {snap})")


def _keep_awake():
    """Windows: keep the machine from sleeping while `watch` runs (reverts
    automatically when the process exits). Closing the lid may still sleep."""
    if sys.platform == "win32":
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        print("keep-awake on: this PC will not sleep while watch runs", flush=True)


def cmd_watch(a):
    """Poll over SSH until the step writes /workspace/exit_code (the pod then
    idles for its grace period), fetch outputs, delete the pod. The launch
    deadline is also enforced from this side."""
    _keep_awake()
    env = _env()
    client = RunPodClient(env["RUNPOD_API_KEY"])
    rec = json.loads(_pod_record(a.pod).read_text()) if _pod_record(a.pod).exists() else {}
    while True:
        try:
            pod = client.get_pod(a.pod)
        except Exception as e:  # transient API/network failure: keep watching
            print(time.strftime("%H:%M:%S"), f"API error ({type(e).__name__}); retrying", flush=True)
            time.sleep(a.every)
            continue
        status = pod.get("desiredStatus")
        ep = ssh_endpoint(pod)
        finished = False
        if status == "RUNNING" and ep:
            probe = "cat /workspace/exit_code 2>/dev/null; tail -n 2 /workspace/run.log 2>/dev/null"
            try:  # one failed probe must never end the watch
                r = subprocess.run(ssh_argv(ep, probe, str(KEY)), capture_output=True, timeout=120)
                finished, lines = parse_probe(r.stdout)
            except Exception as e:
                finished, lines = False, [f"probe failed: {type(e).__name__}"]
            tail = " / ".join(lines[-2:])[-160:].encode("ascii", "replace").decode()
            print(time.strftime("%H:%M:%S"), status, "|", tail, flush=True)
        else:
            print(time.strftime("%H:%M:%S"), status, flush=True)
            if status in ("EXITED", "STOPPED", "TERMINATED"):
                sys.exit("pod stopped before outputs were fetched: start it from the RunPod console, then run `fetch`.")
        if finished:
            break
        if rec.get("deadline") and time.time() > rec["deadline"]:
            print("deadline exceeded: fetching what exists and deleting the pod")
            break
        time.sleep(a.every)
    try:
        finish_pod(client, a.pod, _fetch)  # the pod is deleted even if fetching fails
    finally:
        ec = ROOT / "outputs" / "pod_runs" / a.pod / "exit_code"
        code = ec.read_text().strip() if ec.exists() else "unknown"
        print(f"step exit code {code}; pod {a.pod} deleted.")


def cmd_fetch(a):
    _fetch(RunPodClient(_env()["RUNPOD_API_KEY"]), a.pod)


def cmd_delete(a):
    RunPodClient(_env()["RUNPOD_API_KEY"]).delete_pod(a.pod)
    print("deleted")


def cmd_list(a):
    pods = RunPodClient(_env()["RUNPOD_API_KEY"]).list_pods()
    known = {r.stem for r in (ROOT / "outputs" / "pods").glob("*.json")}
    for p in pods:
        mine = " (launched by this project)" if p.get("id") in known else ""
        print(p.get("id"), p.get("name"), p.get("desiredStatus"), p.get("costPerHr"), mine)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("launch")
    s.add_argument("--step", required=True, help='vq.cli arguments, e.g. "pilot --generator qwen2.5-1.5b"')
    s.add_argument("--hours", type=float, required=True, help="hard cap for install + run")
    s.add_argument("--grace", type=int, default=20, help="minutes the pod waits for `watch` before self-stopping")
    s.add_argument("--cloud", default="COMMUNITY", choices=["SECURE", "COMMUNITY"])
    s.add_argument("--extras", default="gpu", help='pip extras; phase 6 needs "gpu,packed"')
    s.add_argument("--force", action="store_true", help="launch a grid step without a passing gate (deliberate)")
    s.add_argument("--watch", action="store_true", help="after launching, watch until done, fetch and delete")
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
