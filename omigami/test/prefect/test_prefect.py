from typing import Dict, Any, List

import pytest

"""
    Learning tests for Prefect:
        - can be used for testing connection to client
        - can be used to learn how prefect client and graphql works
"""


def test_connection(prefect_service):
    prefect_client = prefect_service

    assert prefect_client


def test_get_flows(prefect_service):

    prefect_client = prefect_service
    projects = get_projects(prefect_client)
    flows = get_project_flows(prefect_client, projects[0]["id"])

    assert flows


def get_projects(prefect_client) -> List[str]:
    query = prefect_client.graphql(
        {"query": {'project': {"name", "id"}}}
    )

    return [item for item in query["data"]["project"]]


def get_project_id(prefect_client, project_name: str) -> str:
    query = prefect_client.graphql(
        {"query": {'project(where: {name: {_eq: "%s"}})' % (project_name): {"name", "id"}}}
    )

    try:
        id = query["data"]["project"][0]["id"]
    except (KeyError, IndexError):
        raise ValueError(
            f"Project {project_name} not found in Prefect Server."
        )

    return id


def get_project_flows(prefect_client, project_id) -> Dict[str, Any]:
    query = prefect_client.graphql(
        {
            "query": {
                'flow(where: {project_id: {_eq: "%s"}})'
                % (project_id): {"name", "version", "id"}
            }
        }
    )

    all_flows = query["data"]["flow"]

    flows = {}
    for flow in all_flows:
        n = flow["name"]
        if n not in flows:
            flows[n] = flow
            continue

        up_to_date_flow = flows[n]
        if up_to_date_flow["version"] < flow["version"]:
            flows[n] = flow

    return flows
