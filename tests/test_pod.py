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
    assert body["gpuTypeIds"] == ["NVIDIA GeForce RTX 4090"]
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
