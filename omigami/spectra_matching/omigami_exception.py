class OmigamiException(Exception):
    status_code = 404

    def __init__(self, message, application_error_code, http_status_code):
        Exception.__init__(self)
        self.message = message
        if http_status_code is not None:
            self.status_code = http_status_code
        self.application_error_code = application_error_code

    def to_dict(self):
        res = {
            "status": {
                "status": self.status_code,
                "message": self.message,
                "app_code": self.application_error_code,
            }
        }
        return res
