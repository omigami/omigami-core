from __future__ import annotations

from prefect import Flow, Client


class Spec2VecDeployer:
    def __init__(self, client: Client):
        self._client = client

    def deploy_flow(self, flow: Flow) -> str:
        """TODO"""

        self._client.create_project(self._project_name)

        training_flow_id = self._client.register(
            flow,
            project_name=self._project_name,
        )

        flow_run_id = self._client.create_flow_run(
            flow_id=training_flow_id,
            run_name=f"run {self._project_name}",
        )

        return flow_run_id
