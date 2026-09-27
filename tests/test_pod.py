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


def test_bootstrap_self_stops_and_has_timeout():
    cmd = bootstrap_command("run-phase --phase phase3_main --generator qwen2.5-1.5b", max_hours=40, grace_minutes=60)
    assert "timeout 2400m" in cmd
    assert "timeout 150m" in bootstrap_command("x", max_hours=2.5)  # no truncation to 2 h
    assert "pip install" in cmd and "python -m vq.cli run-phase" in cmd
    # results stay fetchable for a grace period, then the pod stops itself
    assert cmd.index("exit_code") < cmd.index("sleep 3600") < cmd.index("runpodctl stop pod $RUNPOD_POD_ID")


def test_secrets_never_in_repr():
    spec = PodSpec(name="x", env={"HF_TOKEN": "hf_secret"})
    assert "hf_secret" not in repr(spec)
