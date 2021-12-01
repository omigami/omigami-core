from typing import Tuple, Optional, Dict, Any

from prefect import Client, Flow


class FlowDeployer:
    def __init__(self, prefect_client: Client):
        self._prefect_client = prefect_client

    def deploy_flow(
        self, flow: Flow, project_name: str, parameters: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """Creates a Prefect project if it doesn't exist. Registers the flow to this
        project, and triggers a run of the flow.

        Parameters
        ----------
        flow:
            Prefect flow to be deployed
        project_name:
            Name of the prefect project.
        parameters:
            Dictionary of prefect flow parameters.

        Returns
        -------
        Flow ID and Flow Run ID
        """

        self._prefect_client.create_project(project_name)
        flow_id = self._prefect_client.register(flow, project_name=project_name)
        flow_run_id = self._prefect_client.create_flow_run(
            flow_id=flow_id, run_name=f"run {project_name}", parameters=parameters
        )

        return flow_id, flow_run_id
