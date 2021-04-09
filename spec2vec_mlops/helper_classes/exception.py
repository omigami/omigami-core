class EmbeddingMakerError(Exception):
    pass


class DeployingError(Exception):
    pass


class ValidateInputException(Exception):

    status_code = 404

    def __init__(self, message, application_error_code, http_status_code):
        Exception.__init__(self)
        self.message = message
        if http_status_code is not None:
            self.status_code = http_status_code
        self.application_error_code = application_error_code

    def to_dict(self):
        rv = {"status": {"status": self.status_code, "message": self.message,
                         "app_code": self.application_error_code}}
        return rv


class IncorrectInputTypeError(ValidateInputException):
    def __init__(self, message, application_error_code, http_status_code):
        super().__init__(message, application_error_code, http_status_code)


class MandatoryKeyMissingError(ValidateInputException):
    pass


class IncorrectPeaksJsonTypeError(ValidateInputException):
    pass


class IncorrectFloatFieldTypeError(ValidateInputException):
    pass


class IncorrectStringFieldTypeError(ValidateInputException):
    pass
