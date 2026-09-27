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
    assert cmd.index("exit_code") < cmd.index("sleep 3600") < cmd.index("runpodctl stop pod $RUNPOD_POD_ID")


def test_bootstrap_rejects_quotes_in_step():
    with pytest.raises(ValueError):
        bootstrap_command("score --x 'y'", max_hours=1)


def test_billed_hours_include_install_and_grace():
    from vq.pod import billed_hours_cap

    assert billed_hours_cap(max_hours=3, grace_minutes=60) == pytest.approx(3 + 1 + 5 / 60)


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
    assert body["minRAMPerGPU"] >= 32        # 8B / PRM-7B pass through CPU RAM while loading (~16 GB BF16)
    assert body["minDownloadMbps"] >= 500    # ~42 GB of verifiers per phase must not take an hour
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
