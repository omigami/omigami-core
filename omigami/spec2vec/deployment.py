from __future__ import annotations

from prefect import Flow, Client


class Spec2VecDeployer:
    def __init__(self, client: Client):
        self._client = client

    def deploy_flow(self, flow: Flow, project_name: str) -> str:
        """Deploys and runs a prefect flow to

        Parameters
        ----------
        flow
        project_name

        Returns
        -------

        """

        self._client.create_project(project_name)
        training_flow_id = self._client.register(flow, project_name=project_name)
        flow_run_id = self._client.create_flow_run(
            flow_id=training_flow_id, run_name=f"run {project_name}"
        )

        return flow_run_id
