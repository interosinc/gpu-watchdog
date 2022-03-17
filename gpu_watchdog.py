from datetime import datetime
import logging
import logging.config
import os
import subprocess
import time
from typing import Dict

from datadog_api_client.v1 import ApiClient, Configuration
from datadog_api_client.v1.api.metrics_api import MetricsApi
from datadog_api_client.v1.model.metrics_payload import MetricsPayload
from datadog_api_client.v1.model.point import Point
from datadog_api_client.v1.model.series import Series
from kubernetes import client, config


LOG_LEVEL = os.getenv("LOG_LEVEL", None)
SECONDS_BETWEEN_SAMPLES = int(os.getenv("SECONDS_BETWEEN_SAMPLES", 5))
command = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv"]

logging.config.fileConfig("logging.conf")
logger = logging.getLogger()

if LOG_LEVEL:
    logger.setLevel(LOG_LEVEL)
    [handler.setLevel(LOG_LEVEL) for handler in logger.handlers]
logger.info(f"App initialized with LOG_LEVEL={logger.level}")


def current_timestamp() -> float:
    return datetime.now().timestamp()


class Watchdog:
    processes = {
        # "<PID>": {"service": "foo-model", "env": "prod", "version": "0.9.8"}
    }

    def __init__(self, kubernetes_client, datadog_client):
        self.kubernetes_client = kubernetes_client
        self.datadog_client = datadog_client


    def send_to_datadog(self, metadata: Dict[str, str], used_memory: int) -> None:
        # API request to submit metrics to Datadog
        tags = [f"{k}:{v}" for k, v in metadata.labels.items()]
        logger.debug(f"{used_memory=} MiB; {tags=}")
        try:
            body = MetricsPayload(
                series=[
                    Series(
                        metric="kubernetes.gpu.usage",
                        type="gauge",
                        points=[Point([current_timestamp(), float(used_memory)])],
                        tags=tags
                    )
                ]
            )
            response = self.datadog_client.submit_metrics(body=body)
        except Exception as e:
            logger.error(e)


    def update(self, metrics: Dict[str, int]) -> None:
        """Given a dictionary of {PIDs -> Usage (MiB)}, update Datadog appropriately."""
        for pid, used_memory in metrics.items():
            if pid not in self.processes:
                try:
                    self.processes[pid] = self.get_pod_data_from_pid(pid)
                except RuntimeError as e:
                    logger.error(e)
                    continue
            self.send_to_datadog(metadata=self.processes[pid], used_memory=used_memory)

        # purge unused PIDs
        self.processes = {x: self.processes[x] for x in metrics}

    def get_pod_data_from_pid(self, pid: str):
        logger.debug(f"Getting pod data for {pid=}")
        # get containerId from /proc/<PID>/cgroup
        try:
            with open(f"/proc/{pid}/cgroup", "r") as f:
                container_id = f.read().split("/")[-1].replace("\n", "")
        except FileNotFoundError:
            raise RuntimeError(f"File /proc/{pid}/cgroup not found")

        try:
            assert len(container_id) > 1
        except AssertionError:
            raise RuntimeError(f"Container ID not found in /proc/{pid}/cgroup")

        logger.debug(f"Retrieved {container_id=} from /proc/{pid}/cgroup")

        # query kubernetes /pods for containerId
        all_pods = self.kubernetes_client.list_pod_for_all_namespaces().items
        if all_pods is None:
            raise RuntimeException("No Pods returned from Kubernetes API")
        pods_with_containers = [
            pod for pod in all_pods if pod.status.container_statuses != None
        ]
        pods = [
            pod.metadata
            for pod in pods_with_containers
            if f"docker://{container_id}"
            in [container.container_id for container in pod.status.container_statuses]
        ]
        try:
            return pods[0]
        except IndexError:
            raise RuntimeError(f"Pod not found for {container_id=}")


def parse_nvidia_stats(stdout: str) -> Dict[str, int]:
    stats = [x.split(", ") for x in stdout.split("\n")[2:-1]]
    return {x[0]: int(x[1].replace(" MiB", "")) for x in stats}


def main():  # pragma: no cover

    logger.debug("Loading kube cluster config")
    config.load_incluster_config()
    logger.debug("Creating Kubernetes API client")
    kubernetes_client = client.CoreV1Api()

    logger.debug("Creating Datadog client")
    dd_config = Configuration()
    datadog_client = MetricsApi(ApiClient(dd_config))
    logger.info("Starting watchdog")
    watchdog = Watchdog(kubernetes_client, datadog_client)

    while True:
        hello = subprocess.run(command, capture_output=True)
        results = parse_nvidia_stats(hello.stdout.decode("utf-8"))
        logger.debug(f"nvidia-smi returned {len(results)} processes")
        watchdog.update(results)
        time.sleep(SECONDS_BETWEEN_SAMPLES)


if __name__ == "__main__":
    main()
