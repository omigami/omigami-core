from seldon_core.flask_utils import SeldonMicroserviceException


class EmbeddingMakerError(Exception):
    pass


class DeployingError(Exception):
    pass


class InvalidInputException(SeldonMicroserviceException):
    def __init__(self, message, status_code=400, payload=None, reason=""):
        super().__init__(message, status_code, payload, reason)


class IncorrectSpectrumDataTypeException(InvalidInputException):
    pass


class MandatoryKeyMissingException(InvalidInputException):
    pass


class IncorrectPeaksJsonTypeException(InvalidInputException):
    pass


class IncorrectFloatFieldTypeException(InvalidInputException):
    pass


class IncorrectStringFieldTypeException(InvalidInputException):
    pass
