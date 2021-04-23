from prefect import task


@task
def check_condition(success: bool):
    return success
