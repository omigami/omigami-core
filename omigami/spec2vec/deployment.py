from __future__ import annotations

from prefect import Flow, Client


class Spec2VecDeployer:
    def __init__(self, prefect_client: Client):
        self._prefect_client = prefect_client

    def deploy_flow(self, flow: Flow, project_name: str) -> str:
        """TODO"""

        self._prefect_client.create_project(project_name)
        training_flow_id = self._prefect_client.register(
            flow, project_name=project_name
        )
        flow_run_id = self._prefect_client.create_flow_run(
            flow_id=training_flow_id, run_name=f"run {project_name}"
        )

        return flow_run_id
