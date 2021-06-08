from logging import Logger

from gensim.models.callbacks import CallbackAny2Vec


class CustomTrainingProgressLogger(CallbackAny2Vec):
    def __init__(self, num_of_epochs: int, logger: Logger):
        """

        Parameters
        ----------
        num_of_epochs:
            Total number of training epochs.
        """
        self.epoch = 0
        self.num_of_epochs = num_of_epochs
        self.loss = 0
        self.logger = logger

    def on_epoch_end(self, model):
        """Return progress of model training"""
        loss = model.get_latest_training_loss()

        self.logger.info(
            " Epoch " + str(self.epoch + 1) + " of " + str(self.num_of_epochs) + ".",
        )
        self.logger.info(
            "Change in loss after epoch {}: {}".format(self.epoch + 1, loss - self.loss)
        )
        self.epoch += 1
        self.loss = loss
