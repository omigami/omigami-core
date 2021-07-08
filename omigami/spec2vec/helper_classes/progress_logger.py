from logging import Logger

import numpy as np


class TaskProgressLogger:
    """
    Helper class that can be used inside prefect shared_tasks to easily log the progress.
    Expects the Prefect Task's logger, the total number of items being iterated during the task,
    the frequency to log in percentage (e.g. 25) and a message to be displayed before the
    progress percentage. Only works accurately with percentages that can divide 100 perfectly
    (10, 20, 25, 50).
    """

    def __init__(
        self,
        logger: Logger,
        num_of_items: int,
        log_frequency_in_perc: int,
        msg: str,
    ):
        self.logger = logger
        self.log_freq = log_frequency_in_perc
        self.log_freq_in_items = np.ceil(num_of_items / (100 / log_frequency_in_perc))
        self.msg = msg
        self.progress_count = 0

    def log(self, count: int):
        if count % self.log_freq_in_items == 0:
            self.progress_count += 1
            self.logger.info(f"{self.msg}: {self.progress_count * self.log_freq}%.")
