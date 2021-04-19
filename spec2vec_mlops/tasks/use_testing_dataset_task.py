from prefect import task


@task()
def use_testing_dataset_task(testing_dataset_path: str = None):
    return True if testing_dataset_path else False
