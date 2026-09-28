import pytest

from vq.pod import PodSpec, RunPodClient, bootstrap_command, ssh_endpoint, ssh_argv, scp_argv


class FakeResp:
    def __init__(self, status, payload):
        self.status_code, self._p = status, payload

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self):
        self.calls = []

    def request(self, method, url, headers=None, json=None, timeout=None):
        self.calls.append((method, url, headers, json))
        if method == "POST" and url.endswith("/pods"):
            return FakeResp(201, {"id": "pod123", "desiredStatus": "RUNNING"})
        if method == "GET":
            return FakeResp(200, {"id": "pod123", "desiredStatus": "RUNNING", "publicIp": "1.2.3.4",
                                  "portMappings": {"22": 40022}, "costPerHr": 0.69})
        return FakeResp(200, {})


def test_create_pod_request_shape():
    s = FakeSession()
    c = RunPodClient("KEY", session=s)
    spec = PodSpec(name="vq-phase3", env={"HF_TOKEN": "hf_x", "PUBLIC_KEY": "ssh-ed25519 AAA"})
    pod = c.create_pod(spec)
    method, url, headers, body = s.calls[0]
    assert method == "POST" and url == "https://rest.runpod.io/v1/pods"
    assert headers["Authorization"] == "Bearer KEY"
    assert body["gpuTypeIds"] == ["NVIDIA GeForce RTX 4090"]  # one GPU type for the whole study
    assert body["cloudType"] == "COMMUNITY"
    assert body["gpuCount"] == 1 and "22/tcp" in body["ports"]
    assert body["env"]["HF_TOKEN"] == "hf_x"
    assert pod["id"] == "pod123"


def test_lifecycle_endpoints():
    s = FakeSession()
    c = RunPodClient("KEY", session=s)
    c.get_pod("p1")
    c.stop_pod("p1")
    c.delete_pod("p1")
    assert [(m, u.split("/v1")[1]) for m, u, _, _ in s.calls] == [
        ("GET", "/pods/p1"), ("POST", "/pods/p1/stop"), ("DELETE", "/pods/p1")]


def test_ssh_endpoint_from_pod():
    assert ssh_endpoint({"publicIp": "1.2.3.4", "portMappings": {"22": 40022}}) == ("1.2.3.4", 40022)
    assert ssh_endpoint({"publicIp": None, "portMappings": {}}) is None


def test_ssh_and_scp_argv():
    a = ssh_argv(("1.2.3.4", 40022), "echo hi", key="~/.ssh/id_ed25519")
    assert a[0] == "ssh" and "-p" in a and "40022" in a and a[-1] == "echo hi" and "root@1.2.3.4" in a
    b = scp_argv(("1.2.3.4", 40022), "local.tar.gz", "/workspace/", key="k", upload=True)
    assert b[0] == "scp" and "-P" in b and b[-1] == "root@1.2.3.4:/workspace/"


def test_bootstrap_bounds_everything_including_install():
    cmd = bootstrap_command("score --phase phase3_main --generator G", max_hours=2.5, grace_minutes=60)
    # one hard cap (with forced kill) around install + run
    assert "timeout -k 120 150m bash -c" in cmd
    assert cmd.index("timeout -k 120 150m") < cmd.index("pip install") < cmd.index("python -m vq.cli score")
    # failed install stops immediately with its own exit code (no python run)
    assert "|| { echo 97 > /workspace/exit_code; exit 97; }" in cmd
    # outputs stay fetchable for the grace period, then the pod stops itself
    # (runpodctl is unconfigured in the image and SSH sessions lack RUNPOD_* vars)
    assert cmd.index("exit_code") < cmd.index("sleep 3600") < cmd.index("bash scripts/self_stop.sh")  # grace=60 here
    assert "runpodctl" not in cmd


def test_bootstrap_rejects_quotes_in_step():
    with pytest.raises(ValueError):
        bootstrap_command("score --x 'y'", max_hours=1)


def test_billed_hours_include_install_and_grace():
    from vq.pod import billed_hours_cap

    assert billed_hours_cap(max_hours=3, grace_minutes=60) == pytest.approx(3 + 1 + 5 / 60)
    assert billed_hours_cap(max_hours=3) == pytest.approx(3 + 20 / 60 + 5 / 60)  # default grace 20 min


def test_launch_deletes_pod_if_setup_fails():
    from vq.pod import launch_pod

    s = FakeSession()
    c = RunPodClient("KEY", session=s)

    def setup(pod_id, ep):
        raise RuntimeError("scp failed")

    with pytest.raises(RuntimeError):
        launch_pod(c, PodSpec(name="x"), setup, wait_s=0, poll_s=0)
    assert ("DELETE", "https://rest.runpod.io/v1/pods/pod123") in [(m, u) for m, u, _, _ in s.calls]


def test_launch_keeps_pod_when_setup_succeeds():
    from vq.pod import launch_pod

    s = FakeSession()
    pid = launch_pod(RunPodClient("KEY", session=s), PodSpec(name="x"), lambda pod_id, ep: None, wait_s=0, poll_s=0)
    assert pid == "pod123" and not any(m == "DELETE" for m, *_ in s.calls)


@pytest.mark.parametrize("step,needs", [
    ("pilot --generator G", False),
    ("generate --generator G --split dev", False),
    ("score --phase phase3_main --generator G --only sky-4b --dev-only", False),
    ("select-lambda --generator G", False),
    ("export-packed --verifier sky-4b", False),
    ("generate --generator G --split eval", True),
    ("score --phase phase3_main --generator G", True),
    ("score --phase phase5_ch --generator G", True),
])
def test_gate_required_for_paid_grid_steps(step, needs):
    from vq.pod import requires_gate

    assert requires_gate(step) is needs


def test_gate_check(tmp_path):
    from vq.pod import gate_allows

    assert gate_allows(tmp_path / "gate.json") is False
    (tmp_path / "gate.json").write_text('{"go": false}')
    assert gate_allows(tmp_path / "gate.json") is False
    (tmp_path / "gate.json").write_text('{"go": true}')
    assert gate_allows(tmp_path / "gate.json") is True


def test_merge_outputs_never_overwrites_conflicts(tmp_path):
    from vq.pod import merge_outputs

    src, dst = tmp_path / "pod", tmp_path / "local"
    (src / "store" / "scores").mkdir(parents=True)
    (dst / "store" / "scores").mkdir(parents=True)
    (src / "store" / "scores" / "new.parquet").write_bytes(b"new")
    (src / "store" / "scores" / "same.parquet").write_bytes(b"same")
    (dst / "store" / "scores" / "same.parquet").write_bytes(b"same")
    (src / "frozen_lambda.json").write_text('{"sky-4b": 0.5}')
    (dst / "frozen_lambda.json").write_text('{"sky-4b": 2.0}')
    rep = merge_outputs(src, dst)
    assert (dst / "store" / "scores" / "new.parquet").read_bytes() == b"new"
    assert rep["copied"] == ["store/scores/new.parquet"] and rep["identical"] == ["store/scores/same.parquet"]
    assert rep["conflicts"] == ["frozen_lambda.json"]
    assert (dst / "frozen_lambda.json").read_text() == '{"sky-4b": 2.0}'  # local freeze kept


def test_secrets_never_in_repr():
    spec = PodSpec(name="x", env={"HF_TOKEN": "hf_secret"})
    assert "hf_secret" not in repr(spec)


def test_model_caches_live_on_the_sized_volume_not_container_disk():
    body = PodSpec(name="x", env={"HF_TOKEN": "t"}).body()
    # all experiment checkpoints ~55 GB + packed exports + outputs + pip cache must fit
    assert body["volumeInGb"] >= 100 and body["volumeMountPath"] == "/workspace"
    assert body["env"]["HF_HOME"].startswith("/workspace/")
    assert body["env"]["PIP_CACHE_DIR"].startswith("/workspace/")
    assert body["env"]["HF_TOKEN"] == "t"  # caller env preserved


def test_caller_can_override_cache_location():
    body = PodSpec(name="x", env={"HF_HOME": "/workspace/other"}).body()
    assert body["env"]["HF_HOME"] == "/workspace/other"


def test_api_errors_include_server_message_but_never_the_key():
    class ErrResp(FakeResp):
        text = '{"error": "no instances available for NVIDIA GeForce RTX 4090"}'

    class ErrSession(FakeSession):
        def request(self, method, url, headers=None, json=None, timeout=None):
            return ErrResp(500, {})

    c = RunPodClient("SECRET-KEY", session=ErrSession())
    with pytest.raises(RuntimeError) as e:
        c.create_pod(PodSpec(name="x"))
    assert "no instances available" in str(e.value) and "500" in str(e.value)
    assert "SECRET-KEY" not in str(e.value)


class FlakyStockSession(FakeSession):
    """POST /pods fails with 'no instances' `fails` times, then succeeds."""

    def __init__(self, fails):
        super().__init__()
        self.fails = fails

    def request(self, method, url, headers=None, json=None, timeout=None):
        if method == "POST" and url.endswith("/pods") and self.fails > 0:
            self.fails -= 1
            self.calls.append((method, url, headers, json))
            r = FakeResp(500, {})
            r.text = '{"error":"create pod: There are no instances currently available","status":500}'
            return r
        return super().request(method, url, headers, json, timeout)


def test_create_retries_while_no_stock_then_succeeds():
    from vq.pod import create_pod_when_available

    s = FlakyStockSession(fails=2)
    pod = create_pod_when_available(RunPodClient("K", session=s), PodSpec(name="x"), retry_s=0, max_wait_s=60)
    assert pod["id"] == "pod123"
    assert sum(1 for m, u, *_ in s.calls if m == "POST" and u.endswith("/pods")) == 3


def test_create_gives_up_after_max_wait():
    from vq.pod import create_pod_when_available

    s = FlakyStockSession(fails=10**6)
    with pytest.raises(RuntimeError, match="no instances"):
        create_pod_when_available(RunPodClient("K", session=s), PodSpec(name="x"), retry_s=1, max_wait_s=2,
                                  sleep=lambda t: None)


def test_other_api_errors_are_not_retried():
    from vq.pod import create_pod_when_available

    class Bad(FakeSession):
        def request(self, method, url, headers=None, json=None, timeout=None):
            r = FakeResp(400, {})
            r.text = '{"error":"invalid gpuTypeIds"}'
            return r

    with pytest.raises(RuntimeError, match="invalid gpuTypeIds"):
        create_pod_when_available(RunPodClient("K", session=Bad()), PodSpec(name="x"), retry_s=0, max_wait_s=60)


def test_finish_pod_deletes_even_when_fetch_fails():
    """Failure seen in the real pilot: fetch crashed, pod kept billing."""
    from vq.pod import finish_pod

    s = FakeSession()
    c = RunPodClient("K", session=s)

    def failing_fetch(client, pid):
        raise RuntimeError("tar: outputs: Cannot stat")

    with pytest.raises(RuntimeError, match="Cannot stat"):
        finish_pod(c, "pod123", failing_fetch)
    assert ("DELETE", "https://rest.runpod.io/v1/pods/pod123") in [(m, u) for m, u, _, _ in s.calls]


def test_finish_pod_deletes_after_successful_fetch():
    from vq.pod import finish_pod

    s = FakeSession()
    got = finish_pod(RunPodClient("K", session=s), "pod123", lambda client, pid: "fetched")
    assert got == "fetched" and any(m == "DELETE" for m, *_ in s.calls)


def test_remote_archive_command_tolerates_missing_outputs():
    from vq.pod import remote_archive_command

    cmd = remote_archive_command("/workspace/vq")
    assert "mkdir -p outputs" in cmd and cmd.index("mkdir -p outputs") < cmd.index("tar czf")


def test_host_requirements_for_this_study():
    body = PodSpec(name="x").body()
    assert body["supportPublicIp"] is True  # COMMUNITY hosts may have no public IP -> no SSH
    # 8B / PRM-7B need ~16 GB while loading; 32 GB excluded the 31 GB RTX 4090 hosts
    # that were actually available (seen in the RunPod UI on 2026-09-28)
    assert 24 <= body["minRAMPerGPU"] <= 30
    assert body["minDownloadMbps"] >= 200    # ~42 GB in <30 min without excluding the available hosts
    assert body["minVCPUPerGPU"] >= 4


class DroppingSession(FakeSession):
    """POST /pods: the connection drops `drops` times; `created_anyway` means the
    server did create the pod before the connection died (ambiguous failure)."""

    def __init__(self, drops, created_anyway):
        super().__init__()
        self.drops, self.created_anyway, self.pods = drops, created_anyway, []

    def request(self, method, url, headers=None, json=None, timeout=None):
        import requests

        self.calls.append((method, url, headers, json))
        if method == "POST" and url.endswith("/pods"):
            if self.drops > 0:
                self.drops -= 1
                if self.created_anyway:
                    self.pods.append({"id": "ghost1", "name": json["name"], "desiredStatus": "RUNNING"})
                raise requests.exceptions.ConnectionError("Remote end closed connection without response")
            self.pods.append({"id": "pod123", "name": json["name"], "desiredStatus": "RUNNING"})
            return FakeResp(201, {"id": "pod123"})
        if method == "GET" and url.endswith("/pods"):
            return FakeResp(200, list(self.pods))
        return super().request(method, url, headers, json, timeout)


def test_dropped_connection_adopts_pod_created_anyway():
    from vq.pod import create_pod_when_available

    s = DroppingSession(drops=1, created_anyway=True)
    pod = create_pod_when_available(RunPodClient("K", session=s), PodSpec(name="vq-pilot-ab12"), retry_s=0,
                                    max_wait_s=60, sleep=lambda t: None)
    assert pod["id"] == "ghost1"  # no second pod created, no orphan
    assert sum(1 for m, u, *_ in s.calls if m == "POST" and u.endswith("/pods")) == 1


def test_dropped_connection_without_pod_retries_creation():
    from vq.pod import create_pod_when_available

    s = DroppingSession(drops=2, created_anyway=False)
    pod = create_pod_when_available(RunPodClient("K", session=s), PodSpec(name="vq-pilot-cd34"), retry_s=0,
                                    max_wait_s=60, sleep=lambda t: None)
    assert pod["id"] == "pod123"


def test_unique_pod_names():
    from vq.pod import unique_pod_name

    a, b = unique_pod_name("pilot --generator G"), unique_pod_name("pilot --generator G")
    assert a != b and a.startswith("vq-pilot-")


def test_idempotent_calls_retry_transient_network_errors(monkeypatch):
    import time

    import requests

    monkeypatch.setattr(time, "sleep", lambda s: None)

    class OnceDropping(FakeSession):
        dropped = False

        def request(self, method, url, headers=None, json=None, timeout=None):
            if method == "DELETE" and not self.dropped:
                self.dropped = True
                raise requests.exceptions.ConnectionError("reset")
            return super().request(method, url, headers, json, timeout)

    s = OnceDropping()
    RunPodClient("K", session=s).delete_pod("p1")  # must not raise: deleting a pod is too important to skip
    assert [m for m, *_ in s.calls] == ["DELETE"]


def test_displayed_price_depends_on_cloud():
    from vq.pod import hourly_eur

    assert hourly_eur("COMMUNITY") < hourly_eur("SECURE")
    assert hourly_eur("SECURE") >= 0.69  # RTX 4090 SECURE 0.74 USD/h


def test_self_stop_script_uses_pid1_env_and_graphql():
    import subprocess
    from pathlib import Path

    script = Path(__file__).parents[1] / "scripts" / "self_stop.sh"
    text = script.read_text()
    assert ". scripts/pod_env.sh" in text  # RUNPOD_* vars come from PID 1 via pod_env.sh
    assert "api.runpod.io/graphql" in text and "podStop" in text  # REST returns 403 for the pod key
    code = " ".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert "runpodctl" not in code  # never *called* (mentioned only in comments)
    assert subprocess.run(["bash", "-n"], input=text.encode()).returncode == 0  # stdin: no Windows path issues


def test_probe_parsing_survives_progress_bars_and_binary_noise():
    from vq.pod import parse_probe

    raw = "0\nLoading weights:  27%|██▋ | 109/399\n".encode("utf-8") + b"\x8f\xff"
    finished, lines = parse_probe(raw)
    assert finished is True and "Loading weights" in lines[1]
    assert parse_probe(b"Loading weights: \xe2\x96\x88\n") == (False, ["Loading weights: █"])
    assert parse_probe(b"") == (False, [])
    assert parse_probe(None) == (False, [])


def test_bootstrap_loads_pod_env_before_install_and_run():
    """Pod env vars (HF_HOME, HF_TOKEN, PIP_CACHE_DIR, RUNPOD_*) exist only in the
    container's PID 1, not in SSH sessions: the first real pilot cached 30 GB of
    models on the 40 GB container disk and ran without HF_TOKEN."""
    cmd = bootstrap_command("pilot --generator G", max_hours=3)
    assert cmd.index(". scripts/pod_env.sh") < cmd.index("pip install") < cmd.index("python -m vq.cli")


def test_pod_env_script_exports_the_needed_variables():
    import subprocess
    from pathlib import Path

    script = Path(__file__).parents[1] / "scripts" / "pod_env.sh"
    text = script.read_text()
    assert "/proc/1/environ" in text
    for var in ("HF_HOME", "HF_TOKEN", "PIP_CACHE_DIR", "RUNPOD_POD_ID", "RUNPOD_API_KEY"):
        assert var in text
    assert subprocess.run(["bash", "-n"], input=text.encode()).returncode == 0  # stdin: no Windows path issues


def _git(repo, *args):
    import subprocess

    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_bundle_ships_committed_lf_content_not_working_copy(tmp_path):
    """Working copies on Windows carry CRLF; a CRLF shell script breaks on the pod."""
    import tarfile

    from vq.pod import bundle_code

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / ".gitattributes").write_text("* text=auto eol=lf\n")
    (repo / "run.sh").write_bytes(b"export A=1\r\necho ok\r\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "x")
    out = bundle_code(repo, tmp_path / "b.tar.gz", extras=[])
    with tarfile.open(out) as t:
        data = t.extractfile("run.sh").read()
    assert b"\r" not in data and data == b"export A=1\necho ok\n"


def test_bundle_refuses_uncommitted_changes(tmp_path):
    from vq.pod import bundle_code

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "a.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "x")
    (repo / "a.py").write_text("x = 2\n")
    with pytest.raises(RuntimeError, match="uncommitted"):
        bundle_code(repo, tmp_path / "b.tar.gz", extras=[])


def test_merge_updates_run_reports_but_never_frozen_files(tmp_path):
    """The chain read a stale local pilot report (merge kept it on conflict)."""
    from vq.pod import merge_outputs

    src, dst = tmp_path / "pod", tmp_path / "local"
    for d in (src, dst):
        d.mkdir()
    (src / "pilot_G.json").write_text('{"ok": true}')
    (dst / "pilot_G.json").write_text('{"ok": false}')
    (src / "signal.json").write_text("new")
    (dst / "signal.json").write_text("old")
    (src / "frozen_lambda.json").write_text('{"sky-4b": 0.5}')
    (dst / "frozen_lambda.json").write_text('{"sky-4b": 2.0}')
    rep = merge_outputs(src, dst)
    assert (dst / "pilot_G.json").read_text() == '{"ok": true}'
    assert (dst / "signal.json").read_text() == "new"
    assert sorted(rep["updated"]) == ["pilot_G.json", "signal.json"]
    assert rep["conflicts"] == ["frozen_lambda.json"]
    assert (dst / "frozen_lambda.json").read_text() == '{"sky-4b": 2.0}'


def test_host_cuda_matches_the_pinned_stack():
    """vllm==0.30.0 pins torch==2.13.0, built for CUDA 13.0: a host whose driver
    only supports CUDA 12.8 failed phase 1 ("NVIDIA driver too old")."""
    from pathlib import Path

    assert PodSpec(name="x").body()["allowedCudaVersions"] == ["13.0"]
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert '"vllm==0.30.0"' in pyproject  # the version validated by the pilot, never "latest"


def test_bootstrap_records_the_software_environment():
    cmd = bootstrap_command("phase1 --generator G", max_hours=3)
    assert "pip freeze > outputs/env/pip_freeze.txt" in cmd
    assert "nvidia-smi" in cmd
    assert cmd.index("pip freeze") < cmd.index("python -m vq.cli")


def test_merge_quarantines_a_whole_campaign_on_any_conflict(tmp_path):
    """new remote shards must not be copied into a local campaign
    whose fingerprint (or any other file) conflicts."""
    from vq.pod import merge_outputs

    src, dst = tmp_path / "pod", tmp_path / "local"
    camp = "store/scores/G__v__bf16.shards"
    (src / camp).mkdir(parents=True)
    (dst / camp).mkdir(parents=True)
    (src / camp / "_fingerprint.txt").write_text("remote")
    (dst / camp / "_fingerprint.txt").write_text("local")
    (src / camp / "00001.parquet").write_bytes(b"new shard")
    (src / "store/scores").mkdir(parents=True, exist_ok=True)
    (src / "store/scores/G__v__nf4.parquet").write_bytes(b"clean")
    rep = merge_outputs(src, dst)
    assert not (dst / camp / "00001.parquet").exists()          # nothing of the conflicting campaign copied
    assert (dst / "store/scores/G__v__nf4.parquet").exists()     # unrelated campaign still merged
    assert "store/scores/G__v__bf16" in rep["quarantined"]


def test_gpu_power_check_rejects_power_capped_hosts():
    """A COMMUNITY 4090 capped at 150 W (default 450 W) ran at 855 MHz and scored
    ~3x slower than the pilot host."""
    from vq.pod import gpu_power_ok

    ok, why = gpu_power_ok("150.00 W, 450.00 W\n")
    assert not ok and "150" in why
    assert gpu_power_ok("450.00 W, 450.00 W")[0]
    assert gpu_power_ok("400.00 W, 450.00 W")[0]      # mild caps are fine
    assert not gpu_power_ok("garbage")[0]             # unreadable -> reject


def test_launch_retries_on_bad_host_and_deletes_it():
    from vq.pod import BadHost, launch_pod

    s = FakeSession()
    c = RunPodClient("KEY", session=s)
    tries = []

    def setup(pod_id, ep):
        tries.append(pod_id)
        if len(tries) == 1:
            raise BadHost("power capped")

    pid = launch_pod(c, PodSpec(name="x"), setup, wait_s=0, poll_s=0)
    assert pid == "pod123" and len(tries) == 2
    assert sum(1 for m, *_ in s.calls if m == "DELETE") == 1


def test_launch_gives_up_after_too_many_bad_hosts():
    from vq.pod import BadHost, launch_pod

    s = FakeSession()

    def setup(pod_id, ep):
        raise BadHost("power capped")

    with pytest.raises(BadHost):
        launch_pod(RunPodClient("KEY", session=s), PodSpec(name="x"), setup, wait_s=0, poll_s=0, max_bad_hosts=3)
    assert sum(1 for m, *_ in s.calls if m == "DELETE") == 3


class Flaky5xxSession(FakeSession):
    """POST /pods fails once with a generic 500 (seen live: 'Something went wrong')."""

    def __init__(self, created_anyway=False):
        super().__init__()
        self.failed, self.created_anyway = False, created_anyway

    def request(self, method, url, headers=None, json=None, timeout=None):
        if method == "POST" and url.endswith("/pods") and not self.failed:
            self.failed = True
            self.calls.append((method, url, headers, json))
            r = FakeResp(500, {})
            r.text = '{"error":"create pod: Something went wrong. Please try again later.","status":500}'
            return r
        if method == "GET" and url.endswith("/pods"):
            self.calls.append((method, url, headers, json))
            return FakeResp(200, [{"id": "ghost1", "name": "x"}] if self.created_anyway else [])
        return super().request(method, url, headers, json, timeout)


def test_create_retries_generic_server_errors():
    from vq.pod import create_pod_when_available

    s = Flaky5xxSession()
    pod = create_pod_when_available(RunPodClient("K", session=s), PodSpec(name="x"), retry_s=0, max_wait_s=60,
                                    sleep=lambda t: None)
    assert pod["id"] == "pod123"


def test_create_adopts_pod_created_behind_a_server_error():
    from vq.pod import create_pod_when_available

    s = Flaky5xxSession(created_anyway=True)
    pod = create_pod_when_available(RunPodClient("K", session=s), PodSpec(name="x"), retry_s=0, max_wait_s=60,
                                    sleep=lambda t: None)
    assert pod["id"] == "ghost1"
