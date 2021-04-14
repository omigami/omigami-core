from seldon_core.flask_utils import SeldonMicroserviceException


class EmbeddingMakerError(Exception):
    pass


class DeployingError(Exception):
    pass


class ValidateInputException(SeldonMicroserviceException):
    def __init__(self, message, status_code=400, payload=None, reason=""):
        super().__init__(message, status_code, payload, reason)


class IncorrectInputTypeError(ValidateInputException):
    pass


class MandatoryKeyMissingError(ValidateInputException):
    pass


class IncorrectPeaksJsonTypeError(ValidateInputException):
    pass


class IncorrectFloatFieldTypeError(ValidateInputException):
    pass


class IncorrectStringFieldTypeError(ValidateInputException):
    pass
