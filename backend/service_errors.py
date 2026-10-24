"""Stable public error codes, separate from confidential upstream diagnostics."""
class ServiceError(Exception):
    def __init__(self, code, status, message, retryable=False):
        super().__init__(code)
        self.code, self.status, self.message, self.retryable = code, status, message, retryable

    def detail(self, request_id):
        return {'code': self.code, 'message': self.message, 'retryable': self.retryable, 'request_id': request_id}
