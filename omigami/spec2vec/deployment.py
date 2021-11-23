from __future__ import annotations

from typing import Tuple

from prefect import Flow, Client


class Spec2VecDeployer:
    def __init__(self, client: Client):
        self._client = client

    def deploy_flow(self, flow: Flow, project_name: str) -> Tuple[str, str]:
        """Creates a Prefect project if it doesn't exist. Registers the flow to this
        project, and triggers a run of the flow.

        Parameters
        ----------
        flow:
            Prefect flow to be deployed
        project_name:
            Name of the prefect project.

        Returns
        -------
        Flow ID and Flow Run ID
        """

        self._client.create_project(project_name)
        flow_id = self._client.register(flow, project_name=project_name)
        flow_run_id = self._client.create_flow_run(
            flow_id=flow_id, run_name=f"run {project_name}"
        )

        return flow_id, flow_run_id
