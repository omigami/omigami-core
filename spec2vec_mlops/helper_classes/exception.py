class EmbeddingMakerError(Exception):
    pass


class DeployingError(Exception):
    pass


class ValidateInputError(Exception):
    pass


class IncorrectInputTypeError(ValidateInputError):
    pass


class MandatoryKeyMissingError(ValidateInputError):
    pass


class IncorrectPeaksJsonTypeError(ValidateInputError):
    pass


class IncorrectFloatFieldTypeError(ValidateInputError):
    pass


class IncorrectStringFieldTypeError(ValidateInputError):
    pass
