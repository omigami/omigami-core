from prefect import Task


class DummyTask(Task):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def run(self, arg1=None, arg2=None, arg3=None, arg4=None, **kwargs) -> None:
        pass
