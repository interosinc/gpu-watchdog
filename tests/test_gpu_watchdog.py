from datetime import datetime
import pytest
from unittest.mock import MagicMock
from unittest.mock import patch, mock_open, call

from datadog_api_client.v1.api.metrics_api import MetricsApi
from datadog_api_client.v1.model.metrics_payload import MetricsPayload
from datadog_api_client.v1.model.point import Point
from datadog_api_client.v1.model.series import Series

import kubernetes.client

from gpu_watchdog import parse_nvidia_stats
from gpu_watchdog import Watchdog


@pytest.mark.parametrize(
    "output,expected",
    [
        (
            "pid, used_gpu_memory [MiB]\n22165, 25 MiB\n13588, 2015 MiB\n6648, 2239 MiB\n",
            {"13588": 2015, "6648": 2239},
        ),
        ("pid, used_gpu_memory [MiB]\n22165, 25 MiB\n", {}),
        ("pid, used_gpu_memory [MiB]\n", {}),
    ],
)
def test_parse_nvidia_stats(output, expected):
    got = parse_nvidia_stats(output)
    assert got == expected


@pytest.fixture(scope="session")
def watchdog():
    yield Watchdog(kubernetes_client=MagicMock(), datadog_client=MagicMock())


# process PID not found in /proc
def test_watchdog_get_pod_data_from_pid_process_not_found(watchdog, caplog):
    pid = "123"
    with pytest.raises(RuntimeError) as e:
        got = watchdog.get_pod_data_from_pid(pid)
    assert f"File /proc/{pid}/cgroup not found" in e.exconly()


# container ID missing in /proc/{PID}/cgroup
def test_watchdog_get_pod_data_from_pid_malformed_container_id(watchdog, caplog):
    pid = "123"
    with pytest.raises(RuntimeError) as e:
        with patch("builtins.open", mock_open(read_data="///")) as mock_file:
            got = watchdog.get_pod_data_from_pid(pid)
    mock_file.assert_called_with(
        f"/proc/{pid}/cgroup", "r"
    )  # should this go in the happy path test?
    assert f"Container ID not found in /proc/{pid}/cgroup" in e.exconly()


# No Pods returned
def test_watchdog_get_pod_data_from_pid_container_not_found(watchdog, caplog):
    pid = "123"
    container_id = "987654321"
    m = MagicMock()
    with pytest.raises(RuntimeError) as e:
        with patch("builtins.open", mock_open(read_data=f"/{container_id}")) as mock_file:
            with patch.object(
                watchdog, "kubernetes_client", return_value=m
            ) as k8s_api_client:
                type(k8s_api_client.list_pod_for_all_namespaces.return_value).items = None
                got = watchdog.get_pod_data_from_pid(pid)
    assert f"Pod not found for {container_id=}" in e.exconly()


# Pod has no containers
def test_watchdog_get_pod_data_from_pid_pod_has_no_containers(watchdog, caplog):
    pid = "123"
    container_id = "987654321"
    m = MagicMock()
    n = MagicMock()
    with pytest.raises(RuntimeError) as e:
        with patch("builtins.open", mock_open(read_data=f"/{container_id}")) as mock_file:
            with patch.object(
                watchdog, "kubernetes_client", return_value=m
            ) as k8s_api_client:
                n.status.container_statuses = None
                type(k8s_api_client.list_pod_for_all_namespaces.return_value).items = [n]
                got = watchdog.get_pod_data_from_pid(pid)
    assert f"Pod not found for {container_id=}" in e.exconly()



# containerID not found in kubernetes
def test_watchdog_get_pod_data_from_pid_container_not_found(watchdog, caplog):
    pid = "123"
    container_id = "987654321"
    m = MagicMock()
    with pytest.raises(RuntimeError) as e:
        with patch("builtins.open", mock_open(read_data=f"/{container_id}")) as mock_file:
            with patch.object(
                watchdog, "kubernetes_client", return_value=m
            ) as k8s_api_client:
                type(k8s_api_client.list_pod_for_all_namespaces.return_value).items = []
                got = watchdog.get_pod_data_from_pid(pid)
    assert f"Pod not found for {container_id=}" in e.exconly()


# happy path
def test_watchdog_get_pod_data_from_pid(watchdog, caplog):
    pid = "123"
    container_id = "987654321"
    pod = MagicMock()
    container = MagicMock()
    container.container_id = f"docker://{container_id}"
    pod.status.container_statuses = [container]
    pod.metadata = {'env': 'some-namespace', 'app': 'my-app', 'version': '0.0.0'}
    m = MagicMock()
    with patch("builtins.open", mock_open(read_data=f"/{container_id}")) as mock_file:
        with patch.object(
            watchdog, "kubernetes_client", return_value=m
        ) as k8s_api_client:
            type(k8s_api_client.list_pod_for_all_namespaces.return_value).items = [pod]
            got = watchdog.get_pod_data_from_pid(pid)
    assert got == pod.metadata


def test_watchdog_update(watchdog):
    watchdog.processes = {
        "123": {'env': 'some-namespace', 'app': 'my-app', 'version': '0.0.0'},
        "456": {'env': 'some-namespace', 'app': 'deleted-app', 'version': '0.0.0'},
    }
    metrics = {"123": 99, "789": 120}
    with patch.object(watchdog, 'send_to_datadog') as send_to_datadog:
        with patch.object(watchdog, 'get_pod_data_from_pid', return_value={'a': 'b'}) as get_pod_data_from_pid:
            watchdog.update(metrics)
    # ensure updates are sent correctly
    send_to_datadog.assert_has_calls([
        call(metadata={'env': 'some-namespace', 'app': 'my-app', 'version': '0.0.0'}, used_memory=99),
        call(metadata={'a': 'b'}, used_memory=120)
    ])
    # ensure old processes are purged
    assert watchdog.processes == {"123": {'env': 'some-namespace', 'app': 'my-app', 'version': '0.0.0'}, "789": {'a': 'b'}}


def test_watchdog_send_to_datadog(watchdog):
    watchdog.datadog_client = MagicMock()
    metadata=kubernetes.client.models.v1_object_meta.V1ObjectMeta(labels={'a': 'b', 'c': 'd'})
    time_right_now = 1641835881.482141
    example_body = MetricsPayload(
        series=[
            Series(
                metric="kubernetes.gpu.usage",
                type="gauge",
                points=[Point([time_right_now, 2.0])],
                tags=["a:b","c:d"]
            )
        ]
    )
    with patch("gpu_watchdog.current_timestamp") as mock_timestamp:
        with patch.object(watchdog, 'datadog_client') as datadog_client:
            mock_timestamp.return_value = time_right_now
            watchdog.send_to_datadog(metadata, 2)
            datadog_client.submit_metrics.assert_called_with(body=example_body)

