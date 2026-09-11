"""Stable, machine-readable failures. Never turn a failed stage into success."""


class H3CEError(Exception):
    def __init__(self, code: str, message: str, details=None):
        self.code = code
        self.message = message
        self.details = details
        super().__init__(f"{code}: {message}")

    def as_dict(self):
        result = {"error": self.code, "message": self.message}
        if self.details is not None:
            result["details"] = self.details
        return result

